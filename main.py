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

ANNUAL_FTE_SHIFTS = 168
BLOCK_FTE_SHIFTS  = 84
SHIFTS_ACS_MF     = 5
SHIFTS_ACS_MSUN   = 7
SHIFTS_ICU        = 7

BLOCK1_MONTHS = [7, 8, 9, 10, 11, 12]
BLOCK2_MONTHS = [1, 2, 3, 4, 5, 6]

# Block 1 holiday weeks (month, day of holiday itself)
# Solver finds the Mon-Sun week containing this date
BLOCK1_HOLIDAYS = [
    (7,  4),   # July 4th
    (9,  7),   # Labor Day (first Mon Sep — approximated)
    (11, 26),  # Thanksgiving (4th Thu Nov — approximated)
    (12, 25),  # Christmas
    (1,  1),   # New Year's (falls in block end)
]

BLOCK2_HOLIDAYS = [
    (1,  19),  # MLK Day (3rd Mon Jan)
    (2,  16),  # Presidents Day (3rd Mon Feb)
    (4,  5),   # Easter (approximated)
    (5,  25),  # Memorial Day (last Mon May)
]

# ─────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ORbit Solver v12'})


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

        # Tag each surgeon with its index
        for i, s in enumerate(surgeons):
            s['_idx'] = i

        # ── Step 1: Solve service weeks ───────────────────────────
        week_assignments = solve_service_weeks(
            surgeons=surgeons,
            months=months,
            block_number=block_number,
            preferences=preferences,
            prior_totals=prior_totals,
        )

        # ── Step 2: Solve call using week assignments as input ────
        call_assignments = solve_call(
            surgeons=surgeons,
            months=months,
            week_assignments=week_assignments,
            preferences=preferences,
        )

        # ── Step 3: Build final output ────────────────────────────
        result = build_output(
            surgeons=surgeons,
            months=months,
            week_assignments=week_assignments,
            call_assignments=call_assignments,
            block_number=block_number,
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
# WEEK UTILITIES
# ─────────────────────────────────────────────────────────────────

def get_all_weeks_deduped(months):
    """
    Build flat deduplicated week list across the full block.
    Each week tagged to the month containing its Monday.
    """
    seen_starts   = set()
    all_weeks     = []
    week_to_month = []

    for mi, (y, mo) in enumerate(months):
        first_day  = datetime(y, mo, 1)
        dow        = first_day.weekday()
        week_start = first_day - timedelta(days=dow)

        while True:
            week_end     = week_start + timedelta(days=6)
            days_in_month = []
            for offset in range(7):
                d = week_start + timedelta(days=offset)
                if d.year == y and d.month == mo:
                    days_in_month.append(d.day - 1)

            if days_in_month:
                if week_start not in seen_starts:
                    seen_starts.add(week_start)
                    canonical_mi = mi
                    for check_mi, (cy, cmo) in enumerate(months):
                        if week_start.year == cy and week_start.month == cmo:
                            canonical_mi = check_mi
                            break
                    all_weeks.append({
                        'start': week_start,
                        'end':   week_end,
                        'label': f"{week_start.strftime('%b %-d')} - {week_end.strftime('%b %-d')}",
                        'year':  y,
                        'month': mo,
                    })
                    week_to_month.append(canonical_mi)

            week_start += timedelta(days=7)
            if week_start.year > y or (
                    week_start.year == y and week_start.month > mo):
                break

    return all_weeks, week_to_month


def get_holiday_weeks(months, block_number):
    """
    Return a set of week_start dates that are holiday weeks.
    A holiday week is the Mon-Sun week containing the holiday date.
    """
    holidays = BLOCK1_HOLIDAYS if block_number == 1 else BLOCK2_HOLIDAYS
    holiday_weeks = set()
    year = months[0][0]

    for mo, day in holidays:
        # Try current year and next year (for New Year's etc.)
        for y in [year, year + 1]:
            try:
                holiday_date = datetime(y, mo, day)
                dow          = holiday_date.weekday()
                week_start   = holiday_date - timedelta(days=dow)
                # Only include if this week falls within our block
                for (wy, wmo) in months:
                    if week_start.year == wy and week_start.month == wmo:
                        holiday_weeks.add(week_start)
                        break
                    # Also check if the week overlaps a block month
                    for offset in range(7):
                        d = week_start + timedelta(days=offset)
                        if (d.year, d.month) in [(y2, m2) for y2, m2 in months]:
                            holiday_weeks.add(week_start)
                            break
            except Exception:
                pass

    return holiday_weeks


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
    All surgeons with a start_date are inactive before it.
    All surgeons with a departure_date are inactive after it.
    This applies identically to all surgeons — fellows, new hires, everyone.
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


def is_active_for_week(surgeon, week):
    """Active if surgeon is active on the Monday starting the week."""
    return is_active_on_date(surgeon, week['start'])


def is_active_for_month(surgeon, year, month):
    """Month-level active check used for call."""
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


def is_fellow(surgeon):
    return 'fellow' in surgeon.get('name', '').lower()


def get_pref(surgeon):
    return surgeon.get('extra_shift_preference', 'baseline') or 'baseline'


# ─────────────────────────────────────────────────────────────────
# FTE TARGET
# ─────────────────────────────────────────────────────────────────

def compute_block_target(surgeon, block_number, prior_totals, months):
    """
    Every surgeon's block target = 84 x FTE for Block 1.
    Block 2 = (168 x FTE) minus Block 1 actuals.
    Prorated for start/departure dates within the block.

    Examples (Block 1):
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
# PREFERENCE PARSING
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
                                   active_in_week, all_weeks):
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
# SOLVER 1 — SERVICE WEEKS
# ─────────────────────────────────────────────────────────────────

def solve_service_weeks(surgeons, months, block_number, preferences, prior_totals):
    """
    SOLVER 1: Assigns the 5 weekly service roles across the full block.

    Hard rules enforced here:
    - Exactly one surgeon per role per week
    - Eligibility flags strictly respected
    - Active status (start/departure dates) for all surgeons
    - Baseline surgeons hard-capped at their block target (84 x FTE)
    - No consecutive 7-day service weeks (ACS M-Sun, McNair, TSICU, SICU)
    - ACS M-Sun cannot repeat consecutive weeks
    - Fellow rotation: 2 ACS + 1 SICU per 2-month period
    - Fellows cannot share same role same week
    - Willing/seeking surgeons absorb overflow beyond baseline caps

    Soft rules (preferences) handled here:
    - Time off / conference weeks → soft penalty
    - Holiday week preferences → soft penalty (ranked by surgeon priority)

    Call is NOT in this solver. Service weeks only.
    Output: dict mapping wi -> {role -> surgeon_name}
    """
    num_surgeons = len(surgeons)

    all_weeks, week_to_month = get_all_weeks_deduped(months)
    num_all_weeks            = len(all_weeks)

    active_in_week = [
        [is_active_for_week(surgeons[s], all_weeks[wi])
         for s in range(num_surgeons)]
        for wi in range(num_all_weeks)
    ]

    fellow_indices = [s for s in range(num_surgeons) if is_fellow(surgeons[s])]

    # Parse time off and conference preferences
    surgeon_time_off = {}
    for s in range(num_surgeons):
        sid   = surgeons[s].get('id', '')
        prefs = get_surgeon_prefs(sid, preferences)
        y_ref = months[0][0]
        off   = parse_date_list(prefs.get('time_off',    ''), y_ref)
        conf  = parse_date_list(prefs.get('conferences', ''), y_ref)
        surgeon_time_off[s] = off | conf

    # Holiday weeks and surgeon holiday preferences
    holiday_weeks   = get_holiday_weeks(months, block_number)
    surgeon_holidays = {}
    for s in range(num_surgeons):
        sid   = surgeons[s].get('id', '')
        prefs = get_surgeon_prefs(sid, preferences)
        # holidays field expected as comma-separated ranked list
        # e.g. "thanksgiving,christmas,laborday"
        surgeon_holidays[s] = prefs.get('holidays', '') or ''

    # FTE targets
    block_targets = [
        compute_block_target(surgeons[s], block_number, prior_totals, months)
        for s in range(num_surgeons)
    ]
    target_shifts = [max(0, round(block_targets[s])) for s in range(num_surgeons)]

    model = cp_model.CpModel()

    # ── Variables ─────────────────────────────────────────────────
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

    # Per-surgeon total shift count
    surgeon_total = [model.NewIntVar(0, 200, f'tot_{s}') for s in range(num_surgeons)]
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
            model.Add(surgeon_total[s] == sum(terms))
        else:
            model.Add(surgeon_total[s] == 0)

    # ── Hard Constraints ──────────────────────────────────────────

    # H1 — One surgeon per weekly role
    for wi in range(num_all_weeks):
        model.AddExactlyOne(acs_msun[wi])
        model.AddExactlyOne(acs_mf[wi])
        model.AddExactlyOne(mcnair[wi])
        model.AddExactlyOne(tsicu[wi])
        model.AddExactlyOne(sicu[wi])

    # H2 — Eligibility and active status
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

    # H3 — No surgeon in two roles simultaneously
    for wi in range(num_all_weeks):
        for s in range(num_surgeons):
            model.Add(
                acs_msun[wi][s] + acs_mf[wi][s] + mcnair[wi][s] +
                tsicu[wi][s]    + sicu[wi][s] <= 1
            )

    # H4 — HARD: No consecutive 7-day service weeks
    # ACS M-Sun, McNair, TSICU, SICU are all 7-day roles.
    # A surgeon cannot hold any 7-day role in two back-to-back weeks.
    seven_day = [acs_msun, mcnair, tsicu, sicu]
    for wi in range(num_all_weeks - 1):
        for s in range(num_surgeons):
            in_wi  = model.NewBoolVar(f'i7_{wi}_{s}')
            in_wi1 = model.NewBoolVar(f'i7_{wi+1}_{s}')
            model.Add(sum(r[wi][s]   for r in seven_day) >= in_wi)
            model.Add(sum(r[wi][s]   for r in seven_day) <= 4 * in_wi)
            model.Add(sum(r[wi+1][s] for r in seven_day) >= in_wi1)
            model.Add(sum(r[wi+1][s] for r in seven_day) <= 4 * in_wi1)
            model.Add(in_wi + in_wi1 <= 1)

    # H5 — HARD: ACS M-Sun cannot repeat consecutive weeks
    for wi in range(num_all_weeks - 1):
        for s in range(num_surgeons):
            model.Add(acs_msun[wi][s] + acs_msun[wi + 1][s] <= 1)

    # H6 — HARD: Baseline surgeon shift cap = their block target
    # Willing: up to 140% of target to absorb overflow
    # Seeking: up to 180% of target to absorb overflow
    # This is the FTE equity rule — baseline surgeons get exactly
    # their proportional share, overflow goes to willing/seeking only.
    for s in range(num_surgeons):
        t    = target_shifts[s]
        pref = get_pref(surgeons[s])
        if t == 0:
            continue
        if pref == 'baseline':
            model.Add(surgeon_total[s] <= t)
        elif pref == 'willing':
            model.Add(surgeon_total[s] <= round(t * 1.4))
        elif pref == 'seeking':
            model.Add(surgeon_total[s] <= round(t * 1.8))

    # H7 — Fellows cannot share same role same week
    if len(fellow_indices) >= 2:
        for wi in range(num_all_weeks):
            for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                model.Add(sum(role[wi][f] for f in fellow_indices) <= 1)

    # H8 — Fellow rotation: 2 ACS + 1 SICU per 2-month period (prorated)
    two_month_periods = get_two_month_periods(months)
    for period_months in two_month_periods:
        period_set = set((pm[0], pm[1]) for pm in period_months)
        period_wi  = [
            wi for wi, week in enumerate(all_weeks)
            if (week['year'], week['month']) in period_set
        ]
        for f in fellow_indices:
            acs_t, sicu_t = compute_fellow_rotation_target(
                surgeons[f], period_months, months, active_in_week, all_weeks
            )
            if acs_t == 0 and sicu_t == 0:
                continue
            if acs_t > 0 and period_wi:
                model.Add(sum(
                    acs_mf[wi][f] + acs_msun[wi][f] for wi in period_wi
                ) == acs_t)
            if sicu_t > 0 and period_wi:
                model.Add(sum(sicu[wi][f] for wi in period_wi) == sicu_t)

    # ── Objective ─────────────────────────────────────────────────
    obj_terms     = []
    penalty_terms = []

    # Phase A: Get every surgeon as close to their target as possible.
    # Under-target is penalized heavily (hospital owes them work).
    # Over-target for willing/seeking is penalized lightly (they agreed).
    # Baseline over-target is blocked by H6 hard cap.
    for s in range(num_surgeons):
        t    = target_shifts[s]
        pref = get_pref(surgeons[s])
        if t == 0:
            continue

        under = model.NewIntVar(0, 200, f'u_{s}')
        over  = model.NewIntVar(0, 200, f'o_{s}')
        model.Add(under >= t - surgeon_total[s])
        model.Add(over  >= surgeon_total[s] - t)

        # Being under target is bad for everyone equally
        penalty_terms.append(150 * under)

        # Being over target — only willing/seeking can be here due to H6
        if pref == 'willing':
            penalty_terms.append(15 * over)
        elif pref == 'seeking':
            penalty_terms.append(3 * over)

        # Reward being assigned at all (drives filling to target)
        obj_terms.append(50 * surgeon_total[s])

    # Phase B: Soft preference penalties

    # Time off and conference weeks
    for wi, week in enumerate(all_weeks):
        for s in range(num_surgeons):
            if surgeon_time_off[s] and week_overlaps_dates(week, surgeon_time_off[s]):
                for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                    penalty_terms.append(80 * role[wi][s])

    # Holiday week preferences
    # If a surgeon has ranked holidays, block their top preference week
    # as a soft penalty. Weight 60 = strongly avoid but not hard block.
    for wi, week in enumerate(all_weeks):
        if week['start'] in holiday_weeks:
            for s in range(num_surgeons):
                if active_in_week[wi][s]:
                    holidays_pref = surgeon_holidays[s]
                    if holidays_pref:
                        # Any holiday preference means they want this week off
                        # (full ranking system to be implemented in portal Phase 2)
                        for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                            penalty_terms.append(60 * role[wi][s])

    # Set objective
    total_obj = []
    if obj_terms:
        total_obj.append(sum(obj_terms))
    if penalty_terms:
        total_obj.append(-sum(penalty_terms))
    if total_obj:
        model.Maximize(sum(total_obj) if len(total_obj) > 1 else total_obj[0])

    # ── Solve ─────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120.0
    solver.parameters.num_search_workers  = 4

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise Exception(
            f"Service week solver failed: {solver.StatusName(status)}. "
            f"The hard constraints on service weeks cannot all be satisfied. "
            f"Likely cause: not enough willing/seeking surgeon capacity to "
            f"cover all weeks after baseline FTE caps are applied. "
            f"Check that at least some surgeons are marked willing or seeking."
        )

    # ── Extract assignments ───────────────────────────────────────
    # week_assignments[wi] = {role_key: surgeon_name}
    week_assignments = {}
    for wi in range(num_all_weeks):
        week_assignments[wi] = {
            'label':     all_weeks[wi]['label'],
            'start':     all_weeks[wi]['start'],
            'year':      all_weeks[wi]['year'],
            'month':     all_weeks[wi]['month'],
            'month_idx': week_to_month[wi],
        }
        for s in range(num_surgeons):
            name = surgeons[s]['name']
            if solver.Value(acs_msun[wi][s]): week_assignments[wi]['ACS (M-Sun)'] = name
            if solver.Value(acs_mf[wi][s]):   week_assignments[wi]['ACS (M-F)']   = name
            if solver.Value(mcnair[wi][s]):    week_assignments[wi]['McNair ICU']  = name
            if solver.Value(tsicu[wi][s]):     week_assignments[wi]['TSICU']       = name
            if solver.Value(sicu[wi][s]):      week_assignments[wi]['SICU']        = name

    return week_assignments


# ─────────────────────────────────────────────────────────────────
# SOLVER 2 — CALL
# ─────────────────────────────────────────────────────────────────

def solve_call(surgeons, months, week_assignments, preferences):
    """
    SOLVER 2: Assigns one call surgeon per night across all block nights.

    Takes the completed service week schedule as fixed input.
    Knows exactly who is on service each week to enforce call restrictions.

    Hard rules:
    - Exactly one call surgeon per night
    - Eligibility (can_call) strictly respected
    - Active status respected for each night
    - McNair surgeon: no call ANY night that week
    - TSICU/SICU/ACS M-Sun surgeon: no call Mon-Sat, Sunday OK
    - ACS M-F surgeon: no call Mon-Thu, Fri/Sat/Sun OK
    - Fellow max 5 call nights per month (hard cap)

    Soft rules:
    - Weekend call equity (fair share per surgeon)
    - Call day preferences (e.g. Rojas-Khalil Fri/Sat)
    - Avoid nights (from preferences)
    - No more than 3 consecutive call nights

    Output: dict mapping (month_idx, day_0indexed) -> surgeon_name
    """
    num_surgeons = len(surgeons)
    num_months   = len(months)
    month_days   = [monthrange(y, mo)[1] for y, mo in months]

    fellow_indices = [s for s in range(num_surgeons) if is_fellow(surgeons[s])]

    # Parse avoid nights preferences
    surgeon_avoid = {}
    for s in range(num_surgeons):
        sid   = surgeons[s].get('id', '')
        prefs = get_surgeon_prefs(sid, preferences)
        y_ref = months[0][0]
        surgeon_avoid[s] = parse_date_list(prefs.get('avoid_nights', ''), y_ref)

    # Active status per surgeon per month
    active_in_month = [
        [is_active_for_month(surgeons[s], y, mo) for s in range(num_surgeons)]
        for mi, (y, mo) in enumerate(months)
    ]

    # Build a lookup: for each night, which surgeons are on which service role
    # week_assignments is keyed by wi (flat week index)
    # We need: for each (mi, day), what role is each surgeon assigned to?
    # Build night_service[mi][d][s] = role_name or None

    all_weeks_list = sorted(week_assignments.keys())

    night_service = {}
    for mi, (y, mo) in enumerate(months):
        night_service[mi] = {}
        days = month_days[mi]
        for d in range(days):
            night_service[mi][d] = {}
            date_dt = datetime(y, mo, d + 1)
            for wi in all_weeks_list:
                wa       = week_assignments[wi]
                ws       = wa['start']
                we       = ws + timedelta(days=6)
                if ws <= date_dt <= we:
                    for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                        name = wa.get(role)
                        if name:
                            for s in range(num_surgeons):
                                if surgeons[s]['name'] == name:
                                    night_service[mi][d][s] = role
                    break

    model = cp_model.CpModel()

    # Call variables [mi][d][s]
    call = [
        [[model.NewBoolVar(f'c_{mi}_{d}_{s}') for s in range(num_surgeons)]
         for d in range(month_days[mi])]
        for mi in range(num_months)
    ]

    # ── Hard Constraints ──────────────────────────────────────────

    # H1 — Exactly one call surgeon per night
    for mi in range(num_months):
        for d in range(month_days[mi]):
            model.AddExactlyOne(call[mi][d])

    # H2 — Eligibility and active status
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                if not active_in_month[mi][s] or not is_eligible(surgeons[s], 'call'):
                    model.Add(call[mi][d][s] == 0)

    # H3 — Call restrictions based on service week assignment
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            dow     = datetime(y, mo, d + 1).weekday()  # 0=Mon, 6=Sun
            for s in range(num_surgeons):
                role = night_service[mi][d].get(s)
                if role is None:
                    continue  # Not on service this night — no restriction
                if role == 'McNair ICU':
                    # No call any night during McNair week
                    model.Add(call[mi][d][s] == 0)
                elif role in ('TSICU', 'SICU', 'ACS (M-Sun)'):
                    # No call Mon-Sat, Sunday OK
                    if dow <= 5:
                        model.Add(call[mi][d][s] == 0)
                elif role == 'ACS (M-F)':
                    # No call Mon-Thu, Fri/Sat/Sun OK
                    if dow <= 3:
                        model.Add(call[mi][d][s] == 0)

    # H4 — Fellow max call nights per month (hard cap = 5)
    for f in fellow_indices:
        max_call = int(surgeons[f].get('max_call_per_month', 5))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][f] for d in range(month_days[mi])) <= max_call
            )

    # H5 — Max call nights per month per surgeon (from profile)
    for s in range(num_surgeons):
        if s in fellow_indices:
            continue
        max_call = int(surgeons[s].get('max_call_per_month', 8))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][s] for d in range(month_days[mi])) <= max_call
            )

    # ── Objective ─────────────────────────────────────────────────
    obj_terms     = []
    penalty_terms = []

    # Weekend call equity
    weekend_nights = [
        (mi, d)
        for mi, (y, mo) in enumerate(months)
        for d in range(month_days[mi])
        if datetime(y, mo, d + 1).weekday() >= 4  # Fri=4, Sat=5, Sun=6
    ]
    total_weekend      = len(weekend_nights)
    call_eligible_list = [
        s for s in range(num_surgeons)
        if is_eligible(surgeons[s], 'call')
    ]
    fair_wknd = max(1, round(total_weekend / max(1, len(call_eligible_list))))

    surgeon_wknd = [
        model.NewIntVar(0, total_weekend, f'wk_{s}')
        for s in range(num_surgeons)
    ]
    for s in range(num_surgeons):
        wvars = [call[mi][d][s] for mi, d in weekend_nights]
        if wvars:
            model.Add(surgeon_wknd[s] == sum(wvars))
        else:
            model.Add(surgeon_wknd[s] == 0)

    for s in call_eligible_list:
        pref      = get_pref(surgeons[s])
        wknd_over = model.NewIntVar(0, total_weekend, f'wo_{s}')
        wknd_undr = model.NewIntVar(0, total_weekend, f'wu_{s}')
        model.Add(wknd_over >= surgeon_wknd[s] - fair_wknd)
        model.Add(wknd_undr >= fair_wknd - surgeon_wknd[s])
        if pref == 'baseline':
            penalty_terms.append(40 * wknd_over)
            penalty_terms.append(20 * wknd_undr)
        elif pref == 'willing':
            penalty_terms.append(15 * wknd_over)
            penalty_terms.append(30 * wknd_undr)
        else:
            penalty_terms.append(3  * wknd_over)
            penalty_terms.append(40 * wknd_undr)

    # Call day preference (e.g. Rojas-Khalil prefers Fri/Sat)
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            dow = datetime(y, mo, d + 1).weekday()
            for s in range(num_surgeons):
                pref_days = surgeons[s].get('call_day_preference', '') or ''
                if pref_days == 'friday_saturday':
                    if dow in (4, 5):
                        obj_terms.append(3 * call[mi][d][s])
                    else:
                        penalty_terms.append(2 * call[mi][d][s])

    # No more than 3 consecutive call nights
    for mi in range(num_months):
        days = month_days[mi]
        for s in range(num_surgeons):
            for d in range(days - 3):
                run4 = model.NewBoolVar(f'r4_{mi}_{d}_{s}')
                model.AddMinEquality(run4, [
                    call[mi][d][s], call[mi][d+1][s],
                    call[mi][d+2][s], call[mi][d+3][s],
                ])
                penalty_terms.append(25 * run4)

    # Avoid nights soft blocking
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                if surgeon_avoid[s] and day_in_dates(y, mo, d, surgeon_avoid[s]):
                    penalty_terms.append(30 * call[mi][d][s])

    # Set objective
    total_obj = []
    if obj_terms:
        total_obj.append(sum(obj_terms))
    if penalty_terms:
        total_obj.append(-sum(penalty_terms))
    if total_obj:
        model.Maximize(sum(total_obj) if len(total_obj) > 1 else total_obj[0])

    # ── Solve ─────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers  = 4

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise Exception(
            f"Call solver failed: {solver.StatusName(status)}. "
            f"Could not assign call to all nights given service week restrictions. "
            f"Check call eligibility — ensure enough surgeons can take call on "
            f"every night after service week restrictions are applied."
        )

    # ── Extract call assignments ──────────────────────────────────
    # call_assignments[(mi, d)] = surgeon_name
    call_assignments = {}
    for mi in range(num_months):
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                if solver.Value(call[mi][d][s]):
                    call_assignments[(mi, d)] = surgeons[s]['name']

    return call_assignments


# ─────────────────────────────────────────────────────────────────
# BUILD FINAL OUTPUT
# ─────────────────────────────────────────────────────────────────

def build_output(surgeons, months, week_assignments, call_assignments,
                 block_number, prior_totals):
    """
    Combines service week and call assignments into the standard
    output format expected by the Next.js frontend.
    Also computes FTE summaries and runs the validation report.
    """
    num_surgeons = len(surgeons)
    num_months   = len(months)
    month_days   = [monthrange(y, mo)[1] for y, mo in months]

    block_targets = [
        compute_block_target(surgeons[s], block_number, prior_totals, months)
        for s in range(num_surgeons)
    ]
    target_shifts = [max(0, round(block_targets[s])) for s in range(num_surgeons)]

    # Group weeks by month
    months_weeks = {mi: [] for mi in range(num_months)}
    for wi in sorted(week_assignments.keys()):
        mi = week_assignments[wi]['month_idx']
        months_weeks[mi].append(wi)

    result = {}
    for mi, (y, mo) in enumerate(months):
        mk           = f"{y}-{str(mo).zfill(2)}"
        result_weeks = []

        for wi in months_weeks[mi]:
            wa        = week_assignments[wi]
            week_data = {'label': wa['label']}
            for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                if role in wa:
                    week_data[role] = wa[role]
            result_weeks.append(week_data)

        result_nights = {}
        for d in range(month_days[mi]):
            name = call_assignments.get((mi, d))
            if name:
                result_nights[str(d + 1)] = {'Call': name, 'Backup': ''}

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

    # ── Validation Report ─────────────────────────────────────────
    violations = []
    warnings   = []

    all_weeks_flat = []
    for mi in range(num_months):
        mk = f"{months[mi][0]}-{str(months[mi][1]).zfill(2)}"
        all_weeks_flat.extend(result[mk]['weeks'])

    for mi, (y, mo) in enumerate(months):
        mk          = f"{y}-{str(mo).zfill(2)}"
        month_label = datetime(y, mo, 1).strftime('%B %Y')

        # All roles filled
        for w in result[mk]['weeks']:
            for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                if role not in w:
                    violations.append(f"{month_label} {w['label']}: {role} not assigned")

        # All nights covered
        for d in range(month_days[mi]):
            if str(d + 1) not in result[mk]['nights']:
                violations.append(f"{month_label} day {d + 1}: No call surgeon")

        # No surgeon in two roles same week
        for w in result[mk]['weeks']:
            seen = {}
            for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                name = w.get(role)
                if name:
                    if name in seen:
                        violations.append(
                            f"{month_label} {w['label']}: {name} in {seen[name]} and {role}"
                        )
                    seen[name] = role

    # Consecutive 7-day week check
    seven_day_labels = ['ACS (M-Sun)', 'McNair ICU', 'TSICU', 'SICU']
    for i in range(len(all_weeks_flat) - 1):
        w1 = all_weeks_flat[i]
        w2 = all_weeks_flat[i + 1]
        for r1 in seven_day_labels:
            for r2 in seven_day_labels:
                n1 = w1.get(r1)
                n2 = w2.get(r2)
                if n1 and n1 == n2:
                    violations.append(
                        f"Consecutive 7-day weeks: {n1} ({r1} -> {r2})"
                    )

    # ACS M-Sun consecutive
    for i in range(len(all_weeks_flat) - 1):
        n1 = all_weeks_flat[i].get('ACS (M-Sun)')
        n2 = all_weeks_flat[i + 1].get('ACS (M-Sun)')
        if n1 and n1 == n2:
            violations.append(f"ACS M-Sun consecutive: {n1} back-to-back")

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

    # Sunday call -> Monday fresh service start (warning only)
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            dow = datetime(y, mo, d + 1).weekday()
            if dow != 6:  # Only check Sundays
                continue
            call_name = result[f"{y}-{str(mo).zfill(2)}"]['nights'].get(
                str(d + 1), {}
            ).get('Call', '')
            if not call_name:
                continue
            # Check if this surgeon starts fresh service the next Monday
            next_day = datetime(y, mo, d + 1) + timedelta(days=1)
            for wi in sorted(week_assignments.keys()):
                wa = week_assignments[wi]
                if wa['start'] == next_day:
                    for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                        if wa.get(role) == call_name:
                            # Check if they were also in prior week
                            prior_week_wi = wi - 1
                            in_prior = False
                            if prior_week_wi >= 0:
                                pw = week_assignments[prior_week_wi]
                                in_prior = any(
                                    pw.get(r) == call_name
                                    for r in ['ACS (M-Sun)', 'ACS (M-F)',
                                              'McNair ICU', 'TSICU', 'SICU']
                                )
                            if not in_prior:
                                warnings.append(
                                    f"{call_name}: call Sun {next_day.strftime('%b %-d')} "
                                    f"then fresh {role} Mon — fix manually"
                                )

    # Fellow rotation validation
    two_month_periods = get_two_month_periods(months)
    fellow_indices    = [s for s in range(num_surgeons) if is_fellow(surgeons[s])]

    all_weeks_list = sorted(week_assignments.keys())
    all_weeks      = [week_assignments[wi] for wi in all_weeks_list]
    active_in_week = [
        [is_active_for_week(surgeons[s], all_weeks[wi])
         for s in range(num_surgeons)]
        for wi in range(len(all_weeks))
    ]

    for period_idx, period_months in enumerate(two_month_periods):
        for f in fellow_indices:
            fname       = surgeons[f]['name']
            acs_t, sicu_t = compute_fellow_rotation_target(
                surgeons[f], period_months, months, active_in_week, all_weeks
            )
            acs_count = sicu_count = 0
            for pm in period_months:
                mk = f"{pm[0]}-{str(pm[1]).zfill(2)}"
                if mk not in result:
                    continue
                for w in result[mk]['weeks']:
                    if w.get('ACS (M-F)')   == fname: acs_count  += 1
                    if w.get('ACS (M-Sun)') == fname: acs_count  += 1
                    if w.get('SICU')        == fname: sicu_count += 1
            if acs_t > 0 and acs_count != acs_t:
                violations.append(
                    f"Fellow {fname} period {period_idx + 1}: "
                    f"{acs_count} ACS weeks (expected {acs_t})"
                )
            if sicu_t > 0 and sicu_count != sicu_t:
                violations.append(
                    f"Fellow {fname} period {period_idx + 1}: "
                    f"{sicu_count} SICU weeks (expected {sicu_t})"
                )

    # FTE equity report
    block_fte_summary    = {}
    weekend_call_summary = {}
    weekend_nights       = [
        (mi, d)
        for mi, (y, mo) in enumerate(months)
        for d in range(month_days[mi])
        if datetime(y, mo, d + 1).weekday() >= 4
    ]

    for s in range(num_surgeons):
        name  = surgeons[s]['name']
        pref  = get_pref(surgeons[s])
        total = sum(
            result[f"{y}-{str(mo).zfill(2)}"]['fte_summary'].get(name, 0)
            for y, mo in months
        )
        t     = target_shifts[s]
        delta = total - t

        # Flag baseline surgeons over target as violations
        if pref == 'baseline' and total > t:
            violations.append(
                f"{name} (baseline): served {total} shifts, target {t} — cap exceeded"
            )
        # Flag significant under-target as warning
        elif t > 0 and delta < -7:
            warnings.append(
                f"{name}: served {total} vs target {t} "
                f"(short {abs(delta)}) — insufficient eligible weeks"
            )

        block_fte_summary[name] = {
            'served': total,
            'target': round(block_targets[s], 1),
            'delta':  round(delta, 1),
        }

        # Weekend call count
        count = sum(
            1 for mi, d in weekend_nights
            if call_assignments.get((mi, d)) == name
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
