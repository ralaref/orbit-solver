from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from calendar import monthrange
import os

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

ANNUAL_FTE_SHIFTS     = 168
BLOCK_FTE_SHIFTS      = 84
SHIFTS_ACS_MF         = 5
SHIFTS_ACS_MSUN       = 7
SHIFTS_ICU            = 7
SHIFTS_CALL           = 0
SHIFTS_BACKUP         = 0
MAX_SERVICE_PER_MONTH = 14

BLOCK1_MONTHS = [7, 8, 9, 10, 11, 12]
BLOCK2_MONTHS = [1, 2, 3, 4, 5, 6]

# ─────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ORbit Solver v4'})


# ─────────────────────────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────────────────────────

@app.route('/solve-block', methods=['POST'])
def solve_block():
    try:
        data         = request.json
        surgeons     = data.get('surgeons', [])
        block_number = data.get('block_number', 1)
        start_year   = data.get('start_year')
        preferences  = data.get('preferences', [])
        prior_totals = data.get('prior_totals', {})

        if not surgeons:
            return jsonify({'success': False, 'error': 'No surgeons provided'}), 400
        if not start_year:
            return jsonify({'success': False, 'error': 'start_year required'}), 400

        if block_number == 1:
            months = [(start_year, m) for m in BLOCK1_MONTHS]
        else:
            months = [(start_year + 1, m) for m in BLOCK2_MONTHS]

        result = solve_full_block(
            surgeons=surgeons,
            months=months,
            block_number=block_number,
            preferences=preferences,
            prior_totals=prior_totals,
        )
        return jsonify({'success': True, 'schedule': result})

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


# ─────────────────────────────────────────────────────────────────
# WEEK CALCULATION
# ─────────────────────────────────────────────────────────────────

def get_weeks_for_month(year, month):
    first_day = datetime(year, month, 1)
    dow = first_day.weekday()  # 0=Mon
    week_start = first_day - timedelta(days=dow)
    weeks = []
    while True:
        week_end = week_start + timedelta(days=6)
        days_in_week = []
        for offset in range(7):
            d = week_start + timedelta(days=offset)
            if d.year == year and d.month == month:
                days_in_week.append(d.day - 1)
        if days_in_week:
            weeks.append({
                'start':         week_start,
                'end':           week_end,
                'label':         f"{week_start.strftime('%b %-d')} - {week_end.strftime('%b %-d')}",
                'days_in_month': days_in_week,
                'year':          year,
                'month':         month,
            })
        week_start += timedelta(days=7)
        if week_start.year > year or (
                week_start.year == year and week_start.month > month):
            break
    return weeks


def get_all_weeks_deduped(months):
    """
    Build a flat, deduplicated list of weeks across the full block.
    Weeks that span two months are included only ONCE, tagged to the
    month that contains their Monday (week_start).
    Each week carries its canonical (year, month) from week_start.
    """
    seen_starts = set()
    all_weeks = []
    week_to_month = []

    for mi, (y, mo) in enumerate(months):
        for week in get_weeks_for_month(y, mo):
            ws = week['start']
            if ws in seen_starts:
                continue
            seen_starts.add(ws)
            # Tag the week to the month index that owns its Monday
            canonical_mi = mi
            for check_mi, (cy, cmo) in enumerate(months):
                if ws.year == cy and ws.month == cmo:
                    canonical_mi = check_mi
                    break
            all_weeks.append(week)
            week_to_month.append(canonical_mi)

    return all_weeks, week_to_month


# ─────────────────────────────────────────────────────────────────
# ELIGIBILITY & ACTIVE STATUS
# ─────────────────────────────────────────────────────────────────

def is_eligible(surgeon, role):
    if role in ('acs_msun', 'acs_mf'):
        return bool(surgeon.get('can_acs', False))
    if role == 'mcnair':
        return bool(surgeon.get('covers_mcnair', False))
    if role == 'tsicu':
        return bool(surgeon.get('covers_tsicu', False))
    if role == 'sicu':
        return bool(surgeon.get('covers_sicu', False))
    if role == 'call':
        return bool(surgeon.get('can_call', False))
    return False


def is_active_on_date(surgeon, dt):
    """
    Check if a surgeon is active on a specific date.
    Uses the actual date rather than month-level granularity.
    This fixes the Todd edge-case where a week spans a month boundary.
    """
    start_str = surgeon.get('start_date') or ''
    if start_str:
        try:
            sd = datetime.strptime(start_str[:10], '%Y-%m-%d')
            if dt < sd:
                return False
        except Exception:
            pass

    depart_str = surgeon.get('departure_date') or ''
    if depart_str:
        try:
            dd = datetime.strptime(depart_str[:10], '%Y-%m-%d')
            if dt > dd:
                return False
        except Exception:
            pass

    return True


def is_active(surgeon, year, month):
    """Month-level active check (used for call variables and FTE reporting)."""
    last_day    = monthrange(year, month)[1]
    month_start = datetime(year, month, 1)
    month_end   = datetime(year, month, last_day)

    start_str = surgeon.get('start_date') or ''
    if start_str:
        try:
            sd = datetime.strptime(start_str[:10], '%Y-%m-%d')
            if sd > month_end:
                return False
        except Exception:
            pass

    depart_str = surgeon.get('departure_date') or ''
    if depart_str:
        try:
            dd = datetime.strptime(depart_str[:10], '%Y-%m-%d')
            if dd < month_start:
                return False
        except Exception:
            pass

    return True


def is_active_for_week(surgeon, week):
    """
    FIX for Bug 3 (Todd): A surgeon is active for a week only if they are
    active on the week's Monday (week_start). This prevents a surgeon with
    a start_date of Nov 1 from being assigned a week starting Oct 28.
    """
    return is_active_on_date(surgeon, week['start'])


def is_fellow(surgeon):
    return 'fellow' in surgeon.get('name', '').lower()


# ─────────────────────────────────────────────────────────────────
# FTE TARGET
# ─────────────────────────────────────────────────────────────────

def compute_block_target(surgeon, block_number, prior_totals, months):
    fte    = float(surgeon.get('fte', 1.0))
    annual = ANNUAL_FTE_SHIFTS * fte

    # Block 1: always half the annual target
    # Block 2: annual target minus what was actually served in Block 1
    if block_number == 1:
        block_target = BLOCK_FTE_SHIFTS * fte        # 84 × FTE
    else:
        prior        = float(prior_totals.get(surgeon.get('name', ''), 0))
        block_target = max(0.0, annual - prior)

    # Prorate for start_date within block
    start_str = surgeon.get('start_date') or ''
    if start_str:
        try:
            sd          = datetime.strptime(start_str[:10], '%Y-%m-%d')
            block_start = datetime(months[0][0], months[0][1], 1)
            last        = monthrange(months[-1][0], months[-1][1])[1]
            block_end   = datetime(months[-1][0], months[-1][1], last)
            if sd > block_start:
                total        = (block_end - block_start).days + 1
                active       = max(0, (block_end - sd).days + 1)
                block_target = block_target * (active / total)
        except Exception:
            pass

    # Prorate for departure_date within block
    depart_str = surgeon.get('departure_date') or ''
    if depart_str:
        try:
            dd          = datetime.strptime(depart_str[:10], '%Y-%m-%d')
            block_start = datetime(months[0][0], months[0][1], 1)
            last        = monthrange(months[-1][0], months[-1][1])[1]
            block_end   = datetime(months[-1][0], months[-1][1], last)
            if dd < block_end:
                total        = (block_end - block_start).days + 1
                active       = max(0, (dd - block_start).days + 1)
                block_target = block_target * (active / total)
        except Exception:
            pass

    return max(0.0, block_target)


# ─────────────────────────────────────────────────────────────────
# PREFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────

def get_surgeon_prefs(surgeon_id, preferences):
    for p in preferences:
        if p.get('surgeon_id') == surgeon_id:
            return p
    return {}


def parse_date_list(text, year):
    """Parse comma-separated date ranges into a set of date objects."""
    import re
    from datetime import date
    dates = set()
    if not text:
        return dates

    months_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    for part in [p.strip() for p in text.split(',')]:
        part = part.lower().strip()
        range_match = re.match(r'([a-z]+)\s+(\d+)\s*[-–]\s*(\d+)', part)
        if range_match:
            mon = months_map.get(range_match.group(1)[:3])
            if mon:
                for day in range(int(range_match.group(2)),
                                 int(range_match.group(3)) + 1):
                    try:
                        dates.add(date(year, mon, day))
                    except Exception:
                        pass
            continue
        single_match = re.match(r'([a-z]+)\s+(\d+)', part)
        if single_match:
            mon = months_map.get(single_match.group(1)[:3])
            if mon:
                try:
                    dates.add(date(year, mon, int(single_match.group(2))))
                except Exception:
                    pass

    return dates


def week_overlaps_dates(week, date_set):
    from datetime import date
    for offset in range(7):
        d = week['start'] + timedelta(days=offset)
        if d.date() in date_set:
            return True
    return False


def day_in_dates(year, month, day_0indexed, date_set):
    from datetime import date
    try:
        return date(year, month, day_0indexed + 1) in date_set
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────
# FELLOW ROTATION HELPERS
# ─────────────────────────────────────────────────────────────────

def get_two_month_periods(months):
    """Split 6 months into three 2-month periods."""
    return [[months[i], months[i + 1]] for i in range(0, 6, 2)]


def compute_fellow_rotation_target(fellow, period_months, months,
                                   active_in_week, all_weeks, week_to_month):
    """
    Compute prorated ACS and SICU week targets for a fellow within a
    2-month period based on active weeks (using week-level active check).
    Full period = 2 ACS + 1 SICU.
    """
    f_idx = fellow['_idx']

    # Identify week indices belonging to this period
    period_month_set = set((pm[0], pm[1]) for pm in period_months)
    period_wi = [
        wi for wi, week in enumerate(all_weeks)
        if (week['year'], week['month']) in period_month_set
    ]

    total_weeks  = len(period_wi)
    active_weeks = sum(1 for wi in period_wi if active_in_week[wi][f_idx])

    if total_weeks == 0 or active_weeks == 0:
        return 0, 0

    ratio = active_weeks / total_weeks

    acs_target  = max(0, round(2 * ratio))
    sicu_target = max(0, round(1 * ratio))

    return acs_target, sicu_target


# ─────────────────────────────────────────────────────────────────
# MAIN SOLVER
# ─────────────────────────────────────────────────────────────────

def solve_full_block(surgeons, months, block_number, preferences,
                     prior_totals):

    num_surgeons = len(surgeons)
    num_months   = len(months)

    # Tag each surgeon with its index
    for i, s in enumerate(surgeons):
        s['_idx'] = i

    # ── Per-month structure ───────────────────────────────────────
    month_days = [monthrange(y, mo)[1] for y, mo in months]

    # FIX Bug 2: Use deduplicated flat week list across all 6 months
    all_weeks, week_to_month = get_all_weeks_deduped(months)
    num_all_weeks = len(all_weeks)

    # ── Active status ─────────────────────────────────────────────
    # active_in_month[mi][s] → bool (used for call variables)
    active_in_month = [
        [is_active(surgeons[s], y, mo) for s in range(num_surgeons)]
        for mi, (y, mo) in enumerate(months)
    ]

    # FIX Bug 3: active_in_week uses week-start date, not month tag
    # This ensures Todd (start Nov 1) is NOT assigned the Oct 28 week
    active_in_week = [
        [is_active_for_week(surgeons[s], all_weeks[wi])
         for s in range(num_surgeons)]
        for wi in range(num_all_weeks)
    ]

    # ── Fellows ───────────────────────────────────────────────────
    fellow_indices = [s for s in range(num_surgeons) if is_fellow(surgeons[s])]

    # ── Preferences ───────────────────────────────────────────────
    surgeon_time_off     = {}
    surgeon_avoid_nights = {}

    for s in range(num_surgeons):
        sid   = surgeons[s].get('id', '')
        prefs = get_surgeon_prefs(sid, preferences)
        y_ref = months[0][0]
        off   = parse_date_list(prefs.get('time_off',    ''), y_ref)
        conf  = parse_date_list(prefs.get('conferences', ''), y_ref)
        surgeon_time_off[s]     = off | conf
        surgeon_avoid_nights[s] = parse_date_list(
            prefs.get('avoid_nights', ''), y_ref
        )

    # ── FTE targets ───────────────────────────────────────────────
    block_targets = [
        compute_block_target(surgeons[s], block_number, prior_totals, months)
        for s in range(num_surgeons)
    ]

    # ── OR-Tools model ────────────────────────────────────────────
    model = cp_model.CpModel()

    # Weekly role variables [wi][s]
    acs_msun = [[model.NewBoolVar(f'am_{wi}_{s}')
                 for s in range(num_surgeons)] for wi in range(num_all_weeks)]
    acs_mf   = [[model.NewBoolVar(f'af_{wi}_{s}')
                 for s in range(num_surgeons)] for wi in range(num_all_weeks)]
    mcnair   = [[model.NewBoolVar(f'mn_{wi}_{s}')
                 for s in range(num_surgeons)] for wi in range(num_all_weeks)]
    tsicu    = [[model.NewBoolVar(f'ts_{wi}_{s}')
                 for s in range(num_surgeons)] for wi in range(num_all_weeks)]
    sicu     = [[model.NewBoolVar(f'si_{wi}_{s}')
                 for s in range(num_surgeons)] for wi in range(num_all_weeks)]

    # Nightly call variables [mi][d][s]
    call = [
        [[model.NewBoolVar(f'ca_{mi}_{d}_{s}')
          for s in range(num_surgeons)]
         for d in range(month_days[mi])]
        for mi in range(num_months)
    ]

    # ══════════════════════════════════════════════════════════════
    # HARD CONSTRAINTS
    # ══════════════════════════════════════════════════════════════

    # H1 — Exactly one surgeon per weekly role
    for wi in range(num_all_weeks):
        model.AddExactlyOne(acs_msun[wi])
        model.AddExactlyOne(acs_mf[wi])
        model.AddExactlyOne(mcnair[wi])
        model.AddExactlyOne(tsicu[wi])
        model.AddExactlyOne(sicu[wi])

    # H2 — Exactly one call surgeon per night
    for mi in range(num_months):
        for d in range(month_days[mi]):
            model.AddExactlyOne(call[mi][d])

    # H3 — Eligibility and active status
    for wi in range(num_all_weeks):
        for s in range(num_surgeons):
            inactive = not active_in_week[wi][s]
            if inactive or not is_eligible(surgeons[s], 'acs_msun'):
                model.Add(acs_msun[wi][s] == 0)
            if inactive or not is_eligible(surgeons[s], 'acs_mf'):
                model.Add(acs_mf[wi][s] == 0)
            if inactive or not is_eligible(surgeons[s], 'mcnair'):
                model.Add(mcnair[wi][s] == 0)
            if inactive or not is_eligible(surgeons[s], 'tsicu'):
                model.Add(tsicu[wi][s] == 0)
            if inactive or not is_eligible(surgeons[s], 'sicu'):
                model.Add(sicu[wi][s] == 0)

    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                if (not active_in_month[mi][s]
                        or not is_eligible(surgeons[s], 'call')):
                    model.Add(call[mi][d][s] == 0)

    # H4 — No surgeon in two weekly roles simultaneously
    for wi in range(num_all_weeks):
        for s in range(num_surgeons):
            model.Add(
                acs_msun[wi][s] + acs_mf[wi][s] + mcnair[wi][s] +
                tsicu[wi][s]    + sicu[wi][s] <= 1
            )

    # H5 — ACS M-F and ACS M-Sun always different surgeons (redundant with H4)
    # H6 — McNair, TSICU, SICU always different surgeons (redundant with H4)

    # H7 — Call restrictions by weekly role
    for wi, week in enumerate(all_weeks):
        mi       = week_to_month[wi]
        y, mo    = months[mi]
        wk_start = week['start']

        for offset in range(7):
            day_dt = wk_start + timedelta(days=offset)
            if day_dt.year != y or day_dt.month != mo:
                continue
            d   = day_dt.day - 1
            dow = day_dt.weekday()  # 0=Mon, 6=Sun

            for s in range(num_surgeons):
                # McNair: no call any night that week
                model.Add(mcnair[wi][s] + call[mi][d][s] <= 1)

                # TSICU: no call Mon–Sat (Sunday last resort = soft only)
                if dow <= 5:
                    model.Add(tsicu[wi][s] + call[mi][d][s] <= 1)

                # SICU: no call Mon–Sat
                if dow <= 5:
                    model.Add(sicu[wi][s] + call[mi][d][s] <= 1)

                # ACS M-Sun: no call Mon–Sat
                if dow <= 5:
                    model.Add(acs_msun[wi][s] + call[mi][d][s] <= 1)

                # ACS M-F: no call Mon–Thu
                if dow <= 3:
                    model.Add(acs_mf[wi][s] + call[mi][d][s] <= 1)

    # H8 — Fellows cannot share same role same week
    if len(fellow_indices) >= 2:
        for wi in range(num_all_weeks):
            for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                model.Add(
                    sum(role[wi][f] for f in fellow_indices) <= 1
                )

    # H9 — Fellow rotation: prorated 2 ACS + 1 SICU per 2-month period
    two_month_periods = get_two_month_periods(months)

    for period_idx, period_months in enumerate(two_month_periods):
        period_month_set = set((pm[0], pm[1]) for pm in period_months)
        period_wi = [
            wi for wi, week in enumerate(all_weeks)
            if (week['year'], week['month']) in period_month_set
        ]

        for f in fellow_indices:
            acs_target, sicu_target = compute_fellow_rotation_target(
                surgeons[f], period_months, months,
                active_in_week, all_weeks, week_to_month
            )

            if acs_target == 0 and sicu_target == 0:
                continue

            acs_vars  = [acs_mf[wi][f] + acs_msun[wi][f] for wi in period_wi]
            sicu_vars = [sicu[wi][f] for wi in period_wi]

            if acs_target > 0 and acs_vars:
                model.Add(sum(acs_vars) == acs_target)
            if sicu_target > 0 and sicu_vars:
                model.Add(sum(sicu_vars) == sicu_target)

    # H10 — Fellow max call nights per month (hard cap)
    for f in fellow_indices:
        max_call = int(surgeons[f].get('max_call_per_month', 5))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][f] for d in range(month_days[mi])) <= max_call
            )

    # H11 — No consecutive 7-day service weeks (HARD — last resort is soft below)
    # ACS M-Sun cannot repeat ACS following week (hard per rules)
    for wi in range(num_all_weeks - 1):
        for s in range(num_surgeons):
            model.Add(acs_msun[wi][s] + acs_msun[wi + 1][s] <= 1)

    # ══════════════════════════════════════════════════════════════
    # SOFT CONSTRAINTS
    # ══════════════════════════════════════════════════════════════

    obj_terms     = []
    penalty_terms = []

    # ────────────────────────────────────────────────────────────
    # FIX Bug 4: Fairness — normalized FTE equity objective
    #
    # OLD approach: weight = target * 2, applied per eligible role.
    # Problem: surgeons eligible for 5 roles got 5× the reward per
    # week vs. single-role surgeons (Rojas-Khalil SICU only, etc.)
    # This caused full-eligibility surgeons to dominate the objective.
    #
    # NEW approach:
    #   1. Compute each surgeon's fair share of each role based on
    #      how many eligible surgeons share that role.
    #   2. Track per-surgeon total service shifts across the block.
    #   3. Penalize deviation from target using a quadratic-like
    #      penalty via auxiliary integer variables (over + under).
    #   4. Single-role surgeons get same equity weight as full-role
    #      surgeons — fairness is role-share-normalized.
    # ────────────────────────────────────────────────────────────

    # Count eligible surgeons per role (for normalization)
    roles_list = ['acs_mf', 'acs_msun', 'mcnair', 'tsicu', 'sicu']
    eligible_count = {
        role: max(1, sum(
            1 for s in range(num_surgeons)
            if is_eligible(surgeons[s], role) and block_targets[s] > 0
        ))
        for role in roles_list
    }

    # Per-surgeon total service shifts (integer variable across full block)
    surgeon_total_shifts = [
        model.NewIntVar(0, 400, f'total_shifts_{s}')
        for s in range(num_surgeons)
    ]

    for s in range(num_surgeons):
        shift_expr = []
        for wi in range(num_all_weeks):
            if not active_in_week[wi][s]:
                continue
            shift_expr.append(SHIFTS_ACS_MF   * acs_mf[wi][s])
            shift_expr.append(SHIFTS_ACS_MSUN * acs_msun[wi][s])
            shift_expr.append(SHIFTS_ICU       * mcnair[wi][s])
            shift_expr.append(SHIFTS_ICU       * tsicu[wi][s])
            shift_expr.append(SHIFTS_ICU       * sicu[wi][s])
        if shift_expr:
            model.Add(surgeon_total_shifts[s] == sum(shift_expr))
        else:
            model.Add(surgeon_total_shifts[s] == 0)

    # Over/under target penalty per surgeon
    for s in range(num_surgeons):
        target_int = max(0, round(block_targets[s]))
        if target_int == 0:
            continue

        over  = model.NewIntVar(0, 200, f'over_{s}')
        under = model.NewIntVar(0, 200, f'under_{s}')
        model.Add(over  >= surgeon_total_shifts[s] - target_int)
        model.Add(under >= target_int - surgeon_total_shifts[s])

        pref = surgeons[s].get('extra_shift_preference', 'baseline')

        # Under-target penalty: same for everyone (being short of target is bad)
        penalty_terms.append(30 * under)

        # Over-target penalty: baseline surgeons penalized more heavily
        if pref == 'baseline':
            penalty_terms.append(20 * over)
        elif pref == 'willing':
            penalty_terms.append(8 * over)
        elif pref == 'seeking':
            penalty_terms.append(2 * over)   # seeking surgeons can go over easily

    # Role-share reward: reward assigning surgeons to roles proportionally
    # Weight is inversely proportional to how many surgeons share that role,
    # so single-role surgeons (Rojas-Khalil SICU only) get equivalent
    # reward per assignment as full-eligibility surgeons.
    role_vars = {
        'acs_mf':   acs_mf,
        'acs_msun': acs_msun,
        'mcnair':   mcnair,
        'tsicu':    tsicu,
        'sicu':     sicu,
    }
    for role, rvars in role_vars.items():
        share_weight = max(1, round(100 / eligible_count[role]))
        for wi in range(num_all_weeks):
            for s in range(num_surgeons):
                if active_in_week[wi][s] and is_eligible(surgeons[s], role):
                    obj_terms.append(share_weight * rvars[wi][s])

    # ── S2: Consecutive 7-day service weeks — soft penalty ────────
    seven_day_roles = [acs_msun, mcnair, tsicu, sicu]
    seven_day_names = ['acs_msun', 'mcnair', 'tsicu', 'sicu']

    for wi in range(num_all_weeks - 1):
        for s in range(num_surgeons):
            for r1_idx, r1 in enumerate(seven_day_roles):
                for r2_idx, r2 in enumerate(seven_day_roles):
                    consec = model.NewBoolVar(
                        f'consec_{wi}_{s}_{r1_idx}_{r2_idx}'
                    )
                    model.AddMinEquality(consec, [r1[wi][s], r2[wi + 1][s]])
                    penalty_terms.append(25 * consec)

    # ── S3: Max 14 service shifts per month — soft penalty ────────
    for mi in range(num_months):
        month_wi = [wi for wi in range(num_all_weeks)
                    if week_to_month[wi] == mi]
        for s in range(num_surgeons):
            pref = surgeons[s].get('extra_shift_preference', 'baseline')
            if pref == 'baseline':
                shift_terms = []
                for wi in month_wi:
                    shift_terms.append(SHIFTS_ACS_MF   * acs_mf[wi][s])
                    shift_terms.append(SHIFTS_ACS_MSUN * acs_msun[wi][s])
                    shift_terms.append(SHIFTS_ICU       * mcnair[wi][s])
                    shift_terms.append(SHIFTS_ICU       * tsicu[wi][s])
                    shift_terms.append(SHIFTS_ICU       * sicu[wi][s])
                if shift_terms:
                    over_month = model.NewIntVar(0, 50, f'over_mo_{mi}_{s}')
                    model.Add(
                        over_month >= sum(shift_terms) - MAX_SERVICE_PER_MONTH
                    )
                    penalty_terms.append(15 * over_month)

    # ── S4: Max call nights per month — soft for non-fellows ──────
    for s in range(num_surgeons):
        if s in fellow_indices:
            continue
        max_call = int(surgeons[s].get('max_call_per_month', 8))
        for mi in range(num_months):
            over_call = model.NewIntVar(0, 31, f'overcall_{mi}_{s}')
            model.Add(
                over_call >= sum(
                    call[mi][d][s] for d in range(month_days[mi])
                ) - max_call
            )
            penalty_terms.append(10 * over_call)

    # ── S5: No more than 3 consecutive call nights ────────────────
    for mi in range(num_months):
        days = month_days[mi]
        for s in range(num_surgeons):
            for d in range(days - 3):
                run4 = model.NewBoolVar(f'run4_{mi}_{d}_{s}')
                model.AddMinEquality(run4, [
                    call[mi][d][s],
                    call[mi][d + 1][s],
                    call[mi][d + 2][s],
                    call[mi][d + 3][s],
                ])
                penalty_terms.append(25 * run4)

    # ────────────────────────────────────────────────────────────
    # FIX Bug 5: Weekend call equity
    #
    # OLD approach: flat penalty of 3 per baseline surgeon per
    # weekend night. This was too weak vs. the FTE reward weight,
    # so one surgeon (Al-Aref) dominated all weekend call.
    #
    # NEW approach:
    #   1. Count total weekend nights in the block.
    #   2. Compute fair share of weekend call per eligible surgeon.
    #   3. Track per-surgeon weekend call count (integer variable).
    #   4. Penalize DEVIATION from fair share heavily (over AND under).
    #   5. Retain preference bonuses (seeking/willing/Rojas-Khalil)
    #      as secondary reward but cap their influence.
    # ────────────────────────────────────────────────────────────

    # Collect all weekend nights (Fri=4, Sat=5, Sun=6)
    weekend_nights = []  # list of (mi, d)
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            dow = datetime(y, mo, d + 1).weekday()
            if dow >= 4:
                weekend_nights.append((mi, d))

    total_weekend_nights = len(weekend_nights)
    call_eligible = [
        s for s in range(num_surgeons)
        if is_eligible(surgeons[s], 'call') and block_targets[s] > 0
    ]
    num_call_eligible = max(1, len(call_eligible))
    fair_weekend_share = total_weekend_nights / num_call_eligible

    # Per-surgeon weekend call count
    surgeon_weekend_call = [
        model.NewIntVar(0, total_weekend_nights + 1, f'wknd_call_{s}')
        for s in range(num_surgeons)
    ]
    for s in range(num_surgeons):
        wknd_vars = [call[mi][d][s] for mi, d in weekend_nights]
        if wknd_vars:
            model.Add(surgeon_weekend_call[s] == sum(wknd_vars))
        else:
            model.Add(surgeon_weekend_call[s] == 0)

    # Penalize deviation from fair weekend share
    fair_int = max(0, round(fair_weekend_share))
    for s in call_eligible:
        wknd_over  = model.NewIntVar(0, total_weekend_nights, f'wknd_over_{s}')
        wknd_under = model.NewIntVar(0, total_weekend_nights, f'wknd_under_{s}')
        model.Add(wknd_over  >= surgeon_weekend_call[s] - fair_int)
        model.Add(wknd_under >= fair_int - surgeon_weekend_call[s])

        pref = surgeons[s].get('extra_shift_preference', 'baseline')
        # Heavy penalty for imbalance — heavier than FTE reward
        if pref == 'baseline':
            penalty_terms.append(40 * wknd_over)
            penalty_terms.append(20 * wknd_under)
        elif pref == 'willing':
            penalty_terms.append(20 * wknd_over)
            penalty_terms.append(30 * wknd_under)
        elif pref == 'seeking':
            penalty_terms.append(5  * wknd_over)   # seeking can take more
            penalty_terms.append(40 * wknd_under)

    # Secondary: preference bonuses for weekend call
    for mi, d in weekend_nights:
        for s in range(num_surgeons):
            pref_days = surgeons[s].get('call_day_preference', '') or ''
            dow = datetime(months[mi][0], months[mi][1], d + 1).weekday()
            if pref_days == 'friday_saturday' and dow in (4, 5):
                obj_terms.append(3 * call[mi][d][s])

    # ── S6: Weekday call preference (non-weekend) ─────────────────
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            dow = datetime(y, mo, d + 1).weekday()
            if dow < 4:  # Mon–Thu
                for s in range(num_surgeons):
                    pref_days = surgeons[s].get('call_day_preference', '') or ''
                    if pref_days == 'friday_saturday':
                        # Penalize non-Fri/Sat call for Rojas-Khalil
                        penalty_terms.append(3 * call[mi][d][s])

    # ── S7: Time off / conference soft blocking ───────────────────
    for wi, week in enumerate(all_weeks):
        for s in range(num_surgeons):
            blocked = surgeon_time_off[s]
            if blocked and week_overlaps_dates(week, blocked):
                for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                    penalty_terms.append(50 * role[wi][s])

    # ── S8: Avoid nights soft blocking ────────────────────────────
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                avoid = surgeon_avoid_nights[s]
                if avoid and day_in_dates(y, mo, d, avoid):
                    penalty_terms.append(30 * call[mi][d][s])

    # ── S9: ACS/ICU allocation soft target ────────────────────────
    # Kept but weight reduced vs. fairness objective to avoid overriding equity
    for s in range(num_surgeons):
        acs_alloc = float(surgeons[s].get('acs_allocation', 0.5))
        icu_alloc = float(surgeons[s].get('icu_allocation', 0.5))
        target    = block_targets[s]

        acs_target_shifts = target * acs_alloc
        icu_target_shifts = target * icu_alloc

        acs_weeks_reward = min(int(acs_target_shifts / SHIFTS_ACS_MF) + 1, 10)
        icu_weeks_reward = min(int(icu_target_shifts / SHIFTS_ICU)    + 1, 10)

        for wi in range(num_all_weeks):
            if not active_in_week[wi][s]:
                continue
            if is_eligible(surgeons[s], 'acs_mf'):
                obj_terms.append(acs_weeks_reward * acs_mf[wi][s])
            if is_eligible(surgeons[s], 'acs_msun'):
                obj_terms.append(acs_weeks_reward * acs_msun[wi][s])
            if is_eligible(surgeons[s], 'mcnair'):
                obj_terms.append(icu_weeks_reward * mcnair[wi][s])
            if is_eligible(surgeons[s], 'tsicu'):
                obj_terms.append(icu_weeks_reward * tsicu[wi][s])
            if is_eligible(surgeons[s], 'sicu'):
                obj_terms.append(icu_weeks_reward * sicu[wi][s])

    # ── Set objective ─────────────────────────────────────────────
    all_obj = []
    if obj_terms:
        all_obj.append(sum(obj_terms))
    if penalty_terms:
        all_obj.append(-sum(penalty_terms))

    if all_obj:
        model.Maximize(sum(all_obj) if len(all_obj) > 1 else all_obj[0])

    # ══════════════════════════════════════════════════════════════
    # SOLVE
    # ══════════════════════════════════════════════════════════════

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 90.0
    solver.parameters.num_search_workers  = 4

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise Exception(
            f"No valid schedule found. Status: {solver.StatusName(status)}. "
            f"Check surgeon eligibility — ensure enough surgeons are eligible "
            f"for each role to cover all weeks."
        )

    # ══════════════════════════════════════════════════════════════
    # BUILD OUTPUT
    # ══════════════════════════════════════════════════════════════

    result = {}

    for mi, (y, mo) in enumerate(months):
        mk = f"{y}-{str(mo).zfill(2)}"
        # Only include weeks whose Monday falls in this month
        month_wi_list = [
            wi for wi in range(num_all_weeks)
            if week_to_month[wi] == mi
        ]

        result_weeks = []
        for wi in month_wi_list:
            week_data = {'label': all_weeks[wi]['label']}
            for s in range(num_surgeons):
                name = surgeons[s]['name']
                if solver.Value(acs_msun[wi][s]):
                    week_data['ACS (M-Sun)'] = name
                if solver.Value(acs_mf[wi][s]):
                    week_data['ACS (M-F)'] = name
                if solver.Value(mcnair[wi][s]):
                    week_data['McNair ICU'] = name
                if solver.Value(tsicu[wi][s]):
                    week_data['TSICU'] = name
                if solver.Value(sicu[wi][s]):
                    week_data['SICU'] = name
            result_weeks.append(week_data)

        result_nights = {}
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                if solver.Value(call[mi][d][s]):
                    result_nights[str(d + 1)] = {
                        'Call':   surgeons[s]['name'],
                        'Backup': ''
                    }

        # FTE summary for this month
        fte_summary = {}
        for s in range(num_surgeons):
            name   = surgeons[s]['name']
            shifts = 0
            for w in result_weeks:
                if w.get('ACS (M-F)')   == name: shifts += SHIFTS_ACS_MF
                if w.get('ACS (M-Sun)') == name: shifts += SHIFTS_ACS_MSUN
                for role in ['McNair ICU', 'TSICU', 'SICU']:
                    if w.get(role) == name: shifts += SHIFTS_ICU
            fte_summary[name] = shifts

        result[mk] = {
            'weeks':       result_weeks,
            'nights':      result_nights,
            'fte_summary': fte_summary,
        }

    # ══════════════════════════════════════════════════════════════
    # VALIDATION REPORT
    # ══════════════════════════════════════════════════════════════

    violations = []
    warnings   = []

    for mi, (y, mo) in enumerate(months):
        mk          = f"{y}-{str(mo).zfill(2)}"
        month_data  = result[mk]
        month_label = datetime(y, mo, 1).strftime('%B %Y')

        # All roles filled every week
        for w in month_data['weeks']:
            for role in ['ACS (M-Sun)', 'ACS (M-F)',
                         'McNair ICU', 'TSICU', 'SICU']:
                if role not in w:
                    violations.append(
                        f"{month_label} {w['label']}: {role} not assigned"
                    )

        # All nights covered
        for d in range(month_days[mi]):
            if str(d + 1) not in month_data['nights']:
                violations.append(
                    f"{month_label} day {d + 1}: No call surgeon assigned"
                )

        # No surgeon in two roles same week
        for w in month_data['weeks']:
            seen = {}
            for role in ['ACS (M-Sun)', 'ACS (M-F)',
                         'McNair ICU', 'TSICU', 'SICU']:
                name = w.get(role)
                if name:
                    if name in seen:
                        violations.append(
                            f"{month_label} {w['label']}: "
                            f"{name} in {seen[name]} and {role}"
                        )
                    seen[name] = role

        # Eligibility and active check
        role_key_map = {
            'ACS (M-Sun)': 'acs_msun',
            'ACS (M-F)':   'acs_mf',
            'McNair ICU':  'mcnair',
            'TSICU':       'tsicu',
            'SICU':        'sicu',
        }
        for w in month_data['weeks']:
            for role_label, role_key in role_key_map.items():
                name = w.get(role_label, '')
                if name:
                    for s in range(num_surgeons):
                        if surgeons[s]['name'] == name:
                            if not is_eligible(surgeons[s], role_key):
                                violations.append(
                                    f"{month_label} {w['label']}: "
                                    f"{name} not eligible for {role_label}"
                                )
                            if not active_in_month[mi][s]:
                                violations.append(
                                    f"{month_label} {w['label']}: "
                                    f"{name} not yet active"
                                )

        # Over 14 shifts in a month — warning
        for s in range(num_surgeons):
            name   = surgeons[s]['name']
            shifts = result[mk]['fte_summary'].get(name, 0)
            if shifts > MAX_SERVICE_PER_MONTH:
                warnings.append(
                    f"{month_label}: {name} has {shifts} service shifts "
                    f"(exceeds {MAX_SERVICE_PER_MONTH} — additional compensation applies)"
                )

    # FIX Bug 2: Consecutive week check using deduplicated flat week list
    # Build output weeks in flat order (no duplicates)
    flat_weeks_out = []
    for wi in range(num_all_weeks):
        mi = week_to_month[wi]
        mk = f"{months[mi][0]}-{str(months[mi][1]).zfill(2)}"
        # Find this week's data from result_weeks by label
        for w in result[mk]['weeks']:
            if w['label'] == all_weeks[wi]['label']:
                flat_weeks_out.append(w)
                break

    seven_day_labels = ['ACS (M-Sun)', 'McNair ICU', 'TSICU', 'SICU']
    for i in range(len(flat_weeks_out) - 1):
        w1 = flat_weeks_out[i]
        w2 = flat_weeks_out[i + 1]
        for r1 in seven_day_labels:
            for r2 in seven_day_labels:
                n1 = w1.get(r1)
                n2 = w2.get(r2)
                if n1 and n1 == n2:
                    warnings.append(
                        f"Consecutive 7-day weeks: {n1} "
                        f"({r1} → {r2}) — flagged for review"
                    )

    # Call run check — warn if 4+ consecutive nights
    for mi, (y, mo) in enumerate(months):
        nights = result[f"{y}-{str(mo).zfill(2)}"]['nights']
        days   = month_days[mi]
        for s in range(num_surgeons):
            name = surgeons[s]['name']
            run  = 0
            for d in range(1, days + 1):
                if nights.get(str(d), {}).get('Call') == name:
                    run += 1
                    if run >= 4:
                        warnings.append(
                            f"{datetime(y, mo, 1).strftime('%B %Y')}: "
                            f"{name} has {run}+ consecutive call nights "
                            f"starting day {d - run + 1}"
                        )
                        break
                else:
                    run = 0

    # Fellow rotation validation
    for period_idx, period_months in enumerate(two_month_periods):
        for f in fellow_indices:
            fname = surgeons[f]['name']
            acs_target, sicu_target = compute_fellow_rotation_target(
                surgeons[f], period_months, months,
                active_in_week, all_weeks, week_to_month
            )
            acs_count  = 0
            sicu_count = 0

            for pm in period_months:
                mi = months.index(pm)
                mk = f"{pm[0]}-{str(pm[1]).zfill(2)}"
                for w in result[mk]['weeks']:
                    if w.get('ACS (M-F)')   == fname: acs_count  += 1
                    if w.get('ACS (M-Sun)') == fname: acs_count  += 1
                    if w.get('SICU')        == fname: sicu_count += 1

            if acs_target > 0 and acs_count != acs_target:
                violations.append(
                    f"Fellow {fname} period {period_idx + 1}: "
                    f"{acs_count} ACS weeks (expected {acs_target})"
                )
            if sicu_target > 0 and sicu_count != sicu_target:
                violations.append(
                    f"Fellow {fname} period {period_idx + 1}: "
                    f"{sicu_count} SICU weeks (expected {sicu_target})"
                )

    # Block FTE summary
    block_fte_summary = {}
    for s in range(num_surgeons):
        name  = surgeons[s]['name']
        total = sum(
            result[f"{y}-{str(mo).zfill(2)}"]['fte_summary'].get(name, 0)
            for y, mo in months
        )
        block_fte_summary[name] = {
            'served': total,
            'target': round(block_targets[s], 1),
            'delta':  round(total - block_targets[s], 1),
        }

    # Weekend call summary (informational)
    weekend_call_summary = {}
    for s in range(num_surgeons):
        name = surgeons[s]['name']
        count = sum(
            1 for mi, d in weekend_nights
            if result[f"{months[mi][0]}-{str(months[mi][1]).zfill(2)}"
                      ]['nights'].get(str(d + 1), {}).get('Call') == name
        )
        if count > 0:
            weekend_call_summary[name] = count

    return {
        'months':     result,
        'validation': {
            'violations':           violations,
            'warnings':             warnings,
            'valid':                len(violations) == 0,
            'block_fte_summary':    block_fte_summary,
            'weekend_call_summary': weekend_call_summary,
        }
    }


# ─────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
