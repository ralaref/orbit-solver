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
    return jsonify({'status': 'ok', 'service': 'ORbit Solver v11'})


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
    first_day  = datetime(year, month, 1)
    dow        = first_day.weekday()
    week_start = first_day - timedelta(days=dow)
    weeks      = []
    while True:
        week_end     = week_start + timedelta(days=6)
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
    seen_starts   = set()
    all_weeks     = []
    week_to_month = []
    for mi, (y, mo) in enumerate(months):
        for week in get_weeks_for_month(y, mo):
            ws = week['start']
            if ws in seen_starts:
                continue
            seen_starts.add(ws)
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
    Date-level active check.
    All surgeons with a start_date are treated identically —
    they are inactive before their start_date, active on and after.
    This applies equally to fellows, new hires, and existing staff.
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
    """Month-level active check for call variables."""
    last_day    = monthrange(year, month)[1]
    month_start = datetime(year, month, 1)
    month_end   = datetime(year, month, last_day)
    start_str   = surgeon.get('start_date') or ''
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
    A surgeon is active for a week if they are active on the Monday
    that starts that week. This correctly handles all start dates.
    """
    return is_active_on_date(surgeon, week['start'])


def is_fellow(surgeon):
    return 'fellow' in surgeon.get('name', '').lower()


def get_pref(surgeon):
    return surgeon.get('extra_shift_preference', 'baseline') or 'baseline'


# ─────────────────────────────────────────────────────────────────
# FTE TARGET
# ─────────────────────────────────────────────────────────────────

def compute_block_target(surgeon, block_number, prior_totals, months):
    """
    Block 1 target = 84 x FTE (exactly half of annual 168 x FTE).
    Block 2 target = (168 x FTE) minus Block 1 actuals.

    Prorated for any surgeon whose start_date or departure_date
    falls within the block. This applies to ALL surgeons equally.

    Examples for Block 1:
      1.0 FTE  -> 84 shifts
      0.5 FTE  -> 42 shifts
      0.38 FTE -> 32 shifts
      0.25 FTE -> 21 shifts
      0.16 FTE -> 13 shifts
    """
    fte    = float(surgeon.get('fte', 1.0))
    annual = ANNUAL_FTE_SHIFTS * fte

    if block_number == 1:
        block_target = BLOCK_FTE_SHIFTS * fte
    else:
        prior        = float(prior_totals.get(surgeon.get('name', ''), 0))
        block_target = max(0.0, annual - prior)

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
        part        = part.lower().strip()
        range_match = re.match(r'([a-z]+)\s+(\d+)\s*[-]\s*(\d+)', part)
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
    return [[months[i], months[i + 1]] for i in range(0, 6, 2)]


def compute_fellow_rotation_target(fellow, period_months, months,
                                   active_in_week, all_weeks, week_to_month):
    f_idx      = fellow['_idx']
    period_set = set((pm[0], pm[1]) for pm in period_months)
    period_wi  = [
        wi for wi, week in enumerate(all_weeks)
        if (week['year'], week['month']) in period_set
    ]
    total_weeks  = len(period_wi)
    active_weeks = sum(1 for wi in period_wi if active_in_week[wi][f_idx])
    if total_weeks == 0 or active_weeks == 0:
        return 0, 0
    ratio       = active_weeks / total_weeks
    acs_target  = max(0, round(2 * ratio))
    sicu_target = max(0, round(1 * ratio))
    return acs_target, sicu_target


# ─────────────────────────────────────────────────────────────────
# MAIN SOLVER
# ─────────────────────────────────────────────────────────────────

def solve_full_block(surgeons, months, block_number, preferences, prior_totals):
    """
    RULES-BASED SCHEDULING — v11

    The approach:
    1. Every surgeon has a hard shift target = 84 x FTE for Block 1.
       baseline surgeons: hard upper bound at target — cannot exceed.
       willing/seeking surgeons: can absorb overflow beyond their own target.

    2. The objective has two clear phases built into the weights:
       Phase A — Get every surgeon to their target (under-target penalized heavily)
       Phase B — Route overflow to willing/seeking only (over-target penalized
                 very heavily for baseline, lightly for willing/seeking)

    3. Hard rules enforced as constraints, not penalties:
       - No consecutive 7-day service weeks
       - ACS M-Sun no repeat consecutive weeks
       - Eligibility strictly enforced
       - Call restrictions during service weeks
       - Fellow rotation requirements
       - Start dates for all surgeons (new hires, fellows, departing)
       - Shift totals bounded by target for baseline surgeons

    4. Call assigned equitably as a separate set of constraints.
    """

    num_surgeons = len(surgeons)
    num_months   = len(months)

    for i, s in enumerate(surgeons):
        s['_idx'] = i

    month_days               = [monthrange(y, mo)[1] for y, mo in months]
    all_weeks, week_to_month = get_all_weeks_deduped(months)
    num_all_weeks            = len(all_weeks)

    active_in_month = [
        [is_active(surgeons[s], y, mo) for s in range(num_surgeons)]
        for mi, (y, mo) in enumerate(months)
    ]
    active_in_week = [
        [is_active_for_week(surgeons[s], all_weeks[wi])
         for s in range(num_surgeons)]
        for wi in range(num_all_weeks)
    ]

    fellow_indices = [s for s in range(num_surgeons) if is_fellow(surgeons[s])]

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
            prefs.get('avoid_nights', ''), y_ref)

    block_targets  = [
        compute_block_target(surgeons[s], block_number, prior_totals, months)
        for s in range(num_surgeons)
    ]
    target_shifts  = [max(0, round(block_targets[s])) for s in range(num_surgeons)]

    model = cp_model.CpModel()

    # ── Decision variables ────────────────────────────────────────
    acs_msun = [[model.NewBoolVar(f'am_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_all_weeks)]
    acs_mf   = [[model.NewBoolVar(f'af_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_all_weeks)]
    mcnair   = [[model.NewBoolVar(f'mn_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_all_weeks)]
    tsicu    = [[model.NewBoolVar(f'ts_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_all_weeks)]
    sicu     = [[model.NewBoolVar(f'si_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_all_weeks)]

    call = [
        [[model.NewBoolVar(f'ca_{mi}_{d}_{s}') for s in range(num_surgeons)]
         for d in range(month_days[mi])]
        for mi in range(num_months)
    ]

    # Per-surgeon total service shifts (integer variable)
    # Used to enforce hard upper bounds for baseline surgeons
    surgeon_total_shifts = [
        model.NewIntVar(0, 200, f'total_{s}')
        for s in range(num_surgeons)
    ]

    for s in range(num_surgeons):
        terms = []
        for wi in range(num_all_weeks):
            if not active_in_week[wi][s]:
                continue
            terms.append(SHIFTS_ACS_MF   * acs_mf[wi][s])
            terms.append(SHIFTS_ACS_MSUN * acs_msun[wi][s])
            terms.append(SHIFTS_ICU       * mcnair[wi][s])
            terms.append(SHIFTS_ICU       * tsicu[wi][s])
            terms.append(SHIFTS_ICU       * sicu[wi][s])
        if terms:
            model.Add(surgeon_total_shifts[s] == sum(terms))
        else:
            model.Add(surgeon_total_shifts[s] == 0)

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

    # H3 — Eligibility and active status (weekly roles)
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

    # H3b — Eligibility and active status (call)
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                if not active_in_month[mi][s] or not is_eligible(surgeons[s], 'call'):
                    model.Add(call[mi][d][s] == 0)

    # H4 — No surgeon in two weekly roles simultaneously
    for wi in range(num_all_weeks):
        for s in range(num_surgeons):
            model.Add(
                acs_msun[wi][s] + acs_mf[wi][s] + mcnair[wi][s] +
                tsicu[wi][s]    + sicu[wi][s] <= 1
            )

    # H5 — Call restrictions during service week
    # McNair: no call any night
    # TSICU/SICU/ACS M-Sun: no call Mon-Sat, Sunday OK
    # ACS M-F: no call Mon-Thu, Fri/Sat/Sun OK
    for wi, week in enumerate(all_weeks):
        mi       = week_to_month[wi]
        y, mo    = months[mi]
        wk_start = week['start']
        for offset in range(7):
            day_dt = wk_start + timedelta(days=offset)
            if day_dt.year != y or day_dt.month != mo:
                continue
            d   = day_dt.day - 1
            dow = day_dt.weekday()
            for s in range(num_surgeons):
                model.Add(mcnair[wi][s] + call[mi][d][s] <= 1)
                if dow <= 5:
                    model.Add(tsicu[wi][s]    + call[mi][d][s] <= 1)
                    model.Add(sicu[wi][s]     + call[mi][d][s] <= 1)
                    model.Add(acs_msun[wi][s] + call[mi][d][s] <= 1)
                if dow <= 3:
                    model.Add(acs_mf[wi][s]   + call[mi][d][s] <= 1)

    # H6 — HARD: No consecutive 7-day service weeks for any surgeon
    # A surgeon cannot have any 7-day role (ACS M-Sun, McNair, TSICU, SICU)
    # in two back-to-back weeks. This is a contractual hard rule.
    seven_day_roles = [acs_msun, mcnair, tsicu, sicu]
    for wi in range(num_all_weeks - 1):
        for s in range(num_surgeons):
            # Sum of 7-day roles in week wi
            in_wi = model.NewBoolVar(f'in7_{wi}_{s}')
            model.Add(sum(r[wi][s] for r in seven_day_roles) >= in_wi)
            model.Add(sum(r[wi][s] for r in seven_day_roles) <= 4 * in_wi)
            # Sum of 7-day roles in week wi+1
            in_wi1 = model.NewBoolVar(f'in7_{wi+1}_{s}')
            model.Add(sum(r[wi+1][s] for r in seven_day_roles) >= in_wi1)
            model.Add(sum(r[wi+1][s] for r in seven_day_roles) <= 4 * in_wi1)
            # Cannot have both
            model.Add(in_wi + in_wi1 <= 1)

    # H7 — HARD: ACS M-Sun cannot repeat consecutive weeks
    for wi in range(num_all_weeks - 1):
        for s in range(num_surgeons):
            model.Add(acs_msun[wi][s] + acs_msun[wi + 1][s] <= 1)

    # H8 — HARD: baseline surgeons cannot exceed their block target
    # willing/seeking surgeons have a higher ceiling to absorb overflow
    for s in range(num_surgeons):
        t    = target_shifts[s]
        pref = get_pref(surgeons[s])
        if t == 0:
            continue
        if pref == 'baseline':
            # Hard ceiling at target — no exceptions
            model.Add(surgeon_total_shifts[s] <= t)
        elif pref == 'willing':
            # Can absorb up to 40% overflow
            model.Add(surgeon_total_shifts[s] <= round(t * 1.4))
        elif pref == 'seeking':
            # Can absorb up to 80% overflow
            model.Add(surgeon_total_shifts[s] <= round(t * 1.8))

    # H9 — Fellows cannot share same role same week
    if len(fellow_indices) >= 2:
        for wi in range(num_all_weeks):
            for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                model.Add(sum(role[wi][f] for f in fellow_indices) <= 1)

    # H10 — Fellow rotation: 2 ACS + 1 SICU per 2-month period (prorated)
    two_month_periods = get_two_month_periods(months)
    for period_idx, period_months in enumerate(two_month_periods):
        period_set = set((pm[0], pm[1]) for pm in period_months)
        period_wi  = [
            wi for wi, week in enumerate(all_weeks)
            if (week['year'], week['month']) in period_set
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

    # H11 — Fellow max call nights per month (hard cap)
    for f in fellow_indices:
        max_call = int(surgeons[f].get('max_call_per_month', 5))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][f] for d in range(month_days[mi])) <= max_call
            )

    # ══════════════════════════════════════════════════════════════
    # OBJECTIVE
    #
    # Two-phase objective using integer variables:
    #
    # Phase A — Reward getting every surgeon close to their target.
    #   We reward shifts up to target at high weight (100 per shift).
    #   We penalize being under target at high weight (200 per shift).
    #   This drives the solver to fill every surgeon to their target.
    #
    # Phase B — After targets are met, route overflow to willing/seeking.
    #   Shifts ABOVE target for baseline = forbidden (H8 hard cap).
    #   Shifts ABOVE target for willing = low penalty (10 per shift).
    #   Shifts ABOVE target for seeking = minimal penalty (2 per shift).
    #   This naturally routes overflow to willing/seeking surgeons.
    #
    # Call equity handled separately with weekend fairness reward.
    # ══════════════════════════════════════════════════════════════

    obj_terms     = []
    penalty_terms = []

    for s in range(num_surgeons):
        t    = target_shifts[s]
        pref = get_pref(surgeons[s])
        if t == 0:
            continue

        # Reward shifts up to target
        under = model.NewIntVar(0, 200, f'under_{s}')
        model.Add(under >= t - surgeon_total_shifts[s])
        model.Add(under >= 0)
        penalty_terms.append(200 * under)  # heavy penalty for being short

        # Penalize shifts over target
        over = model.NewIntVar(0, 200, f'over_{s}')
        model.Add(over >= surgeon_total_shifts[s] - t)
        model.Add(over >= 0)

        if pref == 'baseline':
            # H8 prevents this but add objective penalty as backup
            penalty_terms.append(500 * over)
        elif pref == 'willing':
            penalty_terms.append(10 * over)
        elif pref == 'seeking':
            penalty_terms.append(2 * over)

        # Reward each shift up to target (encourages filling to target)
        obj_terms.append(100 * surgeon_total_shifts[s])

    # Weekend call equity
    weekend_nights = [
        (mi, d)
        for mi, (y, mo) in enumerate(months)
        for d in range(month_days[mi])
        if datetime(y, mo, d + 1).weekday() >= 4
    ]
    total_weekend      = len(weekend_nights)
    call_eligible_list = [
        s for s in range(num_surgeons)
        if is_eligible(surgeons[s], 'call') and target_shifts[s] > 0
    ]
    fair_wknd = max(1, round(total_weekend / max(1, len(call_eligible_list))))

    surgeon_weekend_call = [
        model.NewIntVar(0, total_weekend, f'wknd_{s}')
        for s in range(num_surgeons)
    ]
    for s in range(num_surgeons):
        wknd_vars = [call[mi][d][s] for mi, d in weekend_nights]
        if wknd_vars:
            model.Add(surgeon_weekend_call[s] == sum(wknd_vars))
        else:
            model.Add(surgeon_weekend_call[s] == 0)

    for s in call_eligible_list:
        pref       = get_pref(surgeons[s])
        wknd_over  = model.NewIntVar(0, total_weekend, f'wo_{s}')
        wknd_under = model.NewIntVar(0, total_weekend, f'wu_{s}')
        model.Add(wknd_over  >= surgeon_weekend_call[s] - fair_wknd)
        model.Add(wknd_under >= fair_wknd - surgeon_weekend_call[s])
        model.Add(wknd_over  >= 0)
        model.Add(wknd_under >= 0)
        if pref == 'baseline':
            penalty_terms.append(40 * wknd_over)
            penalty_terms.append(20 * wknd_under)
        elif pref == 'willing':
            penalty_terms.append(15 * wknd_over)
            penalty_terms.append(30 * wknd_under)
        else:
            penalty_terms.append(3  * wknd_over)
            penalty_terms.append(40 * wknd_under)

    # Call day preference
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            dow = datetime(y, mo, d + 1).weekday()
            for s in range(num_surgeons):
                pref_days = surgeons[s].get('call_day_preference', '') or ''
                if pref_days == 'friday_saturday' and dow in (4, 5):
                    obj_terms.append(3 * call[mi][d][s])

    # No more than 3 consecutive call nights
    for mi in range(num_months):
        days = month_days[mi]
        for s in range(num_surgeons):
            for d in range(days - 3):
                run4 = model.NewBoolVar(f'run4_{mi}_{d}_{s}')
                model.AddMinEquality(run4, [
                    call[mi][d][s], call[mi][d+1][s],
                    call[mi][d+2][s], call[mi][d+3][s],
                ])
                penalty_terms.append(25 * run4)

    # Time off / conference blocking
    for wi, week in enumerate(all_weeks):
        for s in range(num_surgeons):
            if surgeon_time_off[s] and week_overlaps_dates(week, surgeon_time_off[s]):
                for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                    penalty_terms.append(50 * role[wi][s])

    # Avoid nights blocking
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                if surgeon_avoid_nights[s] and day_in_dates(y, mo, d, surgeon_avoid_nights[s]):
                    penalty_terms.append(30 * call[mi][d][s])

    # Set objective
    total_obj = []
    if obj_terms:
        total_obj.append(sum(obj_terms))
    if penalty_terms:
        total_obj.append(-sum(penalty_terms))
    if total_obj:
        model.Maximize(sum(total_obj) if len(total_obj) > 1 else total_obj[0])

    # ══════════════════════════════════════════════════════════════
    # SOLVE
    # ══════════════════════════════════════════════════════════════

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180.0
    solver.parameters.num_search_workers  = 4

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise Exception(
            f"No valid schedule found. Status: {solver.StatusName(status)}. "
            f"The combination of hard constraints (shift targets, no consecutive "
            f"7-day weeks, eligibility, fellow rotation) cannot all be satisfied "
            f"simultaneously. Check that willing/seeking surgeons have enough "
            f"capacity to cover all weeks after baseline caps are applied."
        )

    # ══════════════════════════════════════════════════════════════
    # BUILD OUTPUT
    # ══════════════════════════════════════════════════════════════

    result = {}
    for mi, (y, mo) in enumerate(months):
        mk            = f"{y}-{str(mo).zfill(2)}"
        month_wi_list = [wi for wi in range(num_all_weeks) if week_to_month[wi] == mi]
        result_weeks  = []
        for wi in month_wi_list:
            week_data = {'label': all_weeks[wi]['label']}
            for s in range(num_surgeons):
                name = surgeons[s]['name']
                if solver.Value(acs_msun[wi][s]): week_data['ACS (M-Sun)'] = name
                if solver.Value(acs_mf[wi][s]):   week_data['ACS (M-F)']   = name
                if solver.Value(mcnair[wi][s]):    week_data['McNair ICU']  = name
                if solver.Value(tsicu[wi][s]):     week_data['TSICU']       = name
                if solver.Value(sicu[wi][s]):      week_data['SICU']        = name
            result_weeks.append(week_data)

        result_nights = {}
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                if solver.Value(call[mi][d][s]):
                    result_nights[str(d + 1)] = {
                        'Call': surgeons[s]['name'], 'Backup': ''
                    }

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
            'weeks': result_weeks,
            'nights': result_nights,
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
            for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                if role not in w:
                    violations.append(f"{month_label} {w['label']}: {role} not assigned")

        # All nights covered
        for d in range(month_days[mi]):
            if str(d + 1) not in month_data['nights']:
                violations.append(f"{month_label} day {d + 1}: No call surgeon assigned")

        # No surgeon in two roles same week
        for w in month_data['weeks']:
            seen = {}
            for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                name = w.get(role)
                if name:
                    if name in seen:
                        violations.append(
                            f"{month_label} {w['label']}: {name} in {seen[name]} and {role}"
                        )
                    seen[name] = role

        # Eligibility check
        role_key_map = {
            'ACS (M-Sun)': 'acs_msun', 'ACS (M-F)': 'acs_mf',
            'McNair ICU': 'mcnair', 'TSICU': 'tsicu', 'SICU': 'sicu',
        }
        for w in month_data['weeks']:
            for role_label, role_key in role_key_map.items():
                name = w.get(role_label, '')
                if name:
                    for s in range(num_surgeons):
                        if surgeons[s]['name'] == name:
                            if not is_eligible(surgeons[s], role_key):
                                violations.append(
                                    f"{month_label} {w['label']}: {name} not eligible for {role_label}"
                                )
                            if not active_in_month[mi][s]:
                                violations.append(
                                    f"{month_label} {w['label']}: {name} not yet active"
                                )

    # Consecutive 7-day week check
    flat_weeks_out = []
    for wi in range(num_all_weeks):
        mi = week_to_month[wi]
        mk = f"{months[mi][0]}-{str(months[mi][1]).zfill(2)}"
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
                    violations.append(
                        f"Consecutive 7-day weeks: {n1} ({r1} -> {r2})"
                    )

    # ACS M-Sun consecutive
    for i in range(len(flat_weeks_out) - 1):
        n1 = flat_weeks_out[i].get('ACS (M-Sun)')
        n2 = flat_weeks_out[i + 1].get('ACS (M-Sun)')
        if n1 and n1 == n2:
            violations.append(f"ACS M-Sun consecutive weeks: {n1} back-to-back")

    # Sunday call -> Monday fresh start (warning only — too complex for hard constraint)
    for wi in range(num_all_weeks):
        week       = all_weeks[wi]
        sun_before = week['start'] - timedelta(days=1)
        sun_mi = None
        sun_d  = None
        for check_mi, (cy, cmo) in enumerate(months):
            if sun_before.year == cy and sun_before.month == cmo:
                sun_mi = check_mi
                sun_d  = sun_before.day - 1
                break
        if sun_mi is None:
            continue
        mk_sun    = f"{months[sun_mi][0]}-{str(months[sun_mi][1]).zfill(2)}"
        call_name = result[mk_sun]['nights'].get(str(sun_d + 1), {}).get('Call', '')
        if not call_name:
            continue
        wk_mi   = week_to_month[wi]
        mk_week = f"{months[wk_mi][0]}-{str(months[wk_mi][1]).zfill(2)}"
        wk_data = next(
            (w for w in result[mk_week]['weeks'] if w['label'] == week['label']), {}
        )
        in_this_week = any(wk_data.get(r) == call_name for r in
                           ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU'])
        if not in_this_week:
            continue
        in_prior_week = False
        if wi > 0:
            prior_wk_mi = week_to_month[wi - 1]
            mk_prior    = f"{months[prior_wk_mi][0]}-{str(months[prior_wk_mi][1]).zfill(2)}"
            prior_data  = next(
                (w for w in result[mk_prior]['weeks']
                 if w['label'] == all_weeks[wi - 1]['label']), {}
            )
            in_prior_week = any(prior_data.get(r) == call_name for r in
                                ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU'])
        if not in_prior_week:
            warnings.append(
                f"{call_name}: call Sun {sun_before.strftime('%b %-d')} "
                f"then fresh service start Mon — fix manually"
            )

    # Call run check
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
            acs_count = sicu_count = 0
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

    # FTE equity check
    block_fte_summary = {}
    for s in range(num_surgeons):
        name  = surgeons[s]['name']
        total = sum(
            result[f"{y}-{str(mo).zfill(2)}"]['fte_summary'].get(name, 0)
            for y, mo in months
        )
        t    = target_shifts[s]
        pref = get_pref(surgeons[s])
        delta = total - t

        if t > 0:
            if pref == 'baseline' and total > t:
                violations.append(
                    f"{name}: served {total} shifts but target is {t} "
                    f"— baseline surgeons cannot exceed their target"
                )
            elif delta < -7:
                warnings.append(
                    f"{name}: served {total} vs target {t} "
                    f"(short by {abs(delta)}) — insufficient eligible weeks available"
                )
            elif delta > 14:
                warnings.append(
                    f"{name}: served {total} vs target {t} "
                    f"(+{delta} over) — absorbing overflow as willing/seeking"
                )

        block_fte_summary[name] = {
            'served': total,
            'target': round(block_targets[s], 1),
            'delta':  round(total - block_targets[s], 1),
        }

    # Weekend call summary
    weekend_call_summary = {}
    for s in range(num_surgeons):
        name  = surgeons[s]['name']
        count = sum(
            1 for mi, d in weekend_nights
            if result[
                f"{months[mi][0]}-{str(months[mi][1]).zfill(2)}"
            ]['nights'].get(str(d + 1), {}).get('Call') == name
        )
        if count > 0:
            weekend_call_summary[name] = count

    return {
        'months': result,
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
