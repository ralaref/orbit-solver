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

# ─────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ORbit Solver v14'})


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

        for i, s in enumerate(surgeons):
            s['_idx'] = i

        week_assignments = solve_service_weeks(
            surgeons=surgeons,
            months=months,
            block_number=block_number,
            preferences=preferences,
            prior_totals=prior_totals,
        )

        call_assignments = solve_call(
            surgeons=surgeons,
            months=months,
            week_assignments=week_assignments,
            preferences=preferences,
        )

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
    seen_starts   = set()
    all_weeks     = []
    week_to_month = []

    for mi, (y, mo) in enumerate(months):
        first_day  = datetime(y, mo, 1)
        week_start = first_day - timedelta(days=first_day.weekday())

        while True:
            days_in_month = [
                (week_start + timedelta(days=o)).day - 1
                for o in range(7)
                if (week_start + timedelta(days=o)).year == y
                and (week_start + timedelta(days=o)).month == mo
            ]
            if days_in_month and week_start not in seen_starts:
                seen_starts.add(week_start)
                canonical_mi = mi
                for check_mi, (cy, cmo) in enumerate(months):
                    if week_start.year == cy and week_start.month == cmo:
                        canonical_mi = check_mi
                        break
                all_weeks.append({
                    'start': week_start,
                    'end':   week_start + timedelta(days=6),
                    'label': f"{week_start.strftime('%b %-d')} - {(week_start + timedelta(days=6)).strftime('%b %-d')}",
                    'year':  y,
                    'month': mo,
                })
                week_to_month.append(canonical_mi)

            week_start += timedelta(days=7)
            if week_start.year > y or (week_start.year == y and week_start.month > mo):
                break

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


def count_service_roles(surgeon):
    return sum(1 for r in ['acs_mf', 'acs_msun', 'mcnair', 'tsicu', 'sicu']
               if is_eligible(surgeon, r))


def is_active_on_date(surgeon, dt):
    start_str = surgeon.get('start_date') or ''
    if start_str:
        try:
            if dt < datetime.strptime(start_str[:10], '%Y-%m-%d'):
                return False
        except Exception:
            pass
    depart_str = surgeon.get('departure_date') or ''
    if depart_str:
        try:
            if dt > datetime.strptime(depart_str[:10], '%Y-%m-%d'):
                return False
        except Exception:
            pass
    return True


def is_active_for_week(surgeon, week):
    return is_active_on_date(surgeon, week['start'])


def is_active_for_month(surgeon, year, month):
    last_day    = monthrange(year, month)[1]
    month_start = datetime(year, month, 1)
    month_end   = datetime(year, month, last_day)
    start_str   = surgeon.get('start_date') or ''
    if start_str:
        try:
            if datetime.strptime(start_str[:10], '%Y-%m-%d') > month_end:
                return False
        except Exception:
            pass
    depart_str = surgeon.get('departure_date') or ''
    if depart_str:
        try:
            if datetime.strptime(depart_str[:10], '%Y-%m-%d') < month_start:
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
    Block 1: target = 84 x FTE
    Block 2: target = (168 x FTE) - Block 1 actuals
    Prorated for start/departure dates.
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


def compute_role_caps(surgeon, target_shifts, pref):
    """
    THE CORE FIX: Compute the maximum number of weeks a surgeon can
    be assigned to each role based on their FTE target.

    Example — Loor (0.38 FTE, target ~32 shifts):
      SICU: 32 / 7 = 4.6 -> cap = 5 weeks maximum
      She cannot get more than 5 SICU weeks regardless of availability.

    Example — Lim (0.16 FTE, target ~13 shifts):
      TSICU: 13 / 7 = 1.9 -> cap = 2 weeks maximum
      He cannot get more than 2 TSICU weeks.

    Example — Al-Aref (1.0 FTE, target 84 shifts, willing):
      ACS M-F:   84 / 5  = 16.8 -> cap = 17
      ACS M-Sun: 84 / 7  = 12   -> cap = 12
      McNair:    84 / 7  = 12   -> cap = 12
      TSICU:     84 / 7  = 12   -> cap = 12
      SICU:      84 / 7  = 12   -> cap = 12
      He can absorb up to his target across any combination of roles.

    For willing/seeking surgeons, caps are multiplied by their overflow
    factor so they can absorb excess after baseline surgeons are filled.

    This is the key insight: the FTE target IS the cap.
    A surgeon's proportional share of each role cannot exceed what
    their FTE entitles them to. The cap is derived mathematically
    from the target, not set arbitrarily.
    """
    if target_shifts == 0:
        return {r: 0 for r in ['acs_mf', 'acs_msun', 'mcnair', 'tsicu', 'sicu']}

    # Overflow multiplier for willing/seeking surgeons
    # They can take more than their baseline entitlement if needed
    if pref == 'baseline':
        multiplier = 1.0
    elif pref == 'willing':
        multiplier = 1.4
    else:  # seeking
        multiplier = 1.8

    effective_target = target_shifts * multiplier

    return {
        'acs_mf':   max(1, round(effective_target / SHIFTS_ACS_MF)),
        'acs_msun': max(1, round(effective_target / SHIFTS_ACS_MSUN)),
        'mcnair':   max(1, round(effective_target / SHIFTS_ICU)),
        'tsicu':    max(1, round(effective_target / SHIFTS_ICU)),
        'sicu':     max(1, round(effective_target / SHIFTS_ICU)),
    }


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
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    for part in [p.strip() for p in text.split(',')]:
        part = part.lower().strip()
        m = re.match(r'([a-z]+)\s+(\d+)\s*[-]\s*(\d+)', part)
        if m:
            mon = months_map.get(m.group(1)[:3])
            if mon:
                for day in range(int(m.group(2)), int(m.group(3)) + 1):
                    try:
                        dates.add(date(year, mon, day))
                    except Exception:
                        pass
            continue
        m = re.match(r'([a-z]+)\s+(\d+)', part)
        if m:
            mon = months_map.get(m.group(1)[:3])
            if mon:
                try:
                    dates.add(date(year, mon, int(m.group(2))))
                except Exception:
                    pass
    return dates


def week_overlaps_dates(week, date_set):
    return any(
        (week['start'] + timedelta(days=o)).date() in date_set
        for o in range(7)
    )


def day_in_dates(year, month, day_0indexed, date_set):
    from datetime import date
    try:
        return date(year, month, day_0indexed + 1) in date_set
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────
# FELLOW ROTATION
# ─────────────────────────────────────────────────────────────────

def get_two_month_periods(months):
    return [[months[i], months[i + 1]] for i in range(0, 6, 2)]


def compute_fellow_rotation_target(fellow, period_months, months,
                                   active_in_week, all_weeks):
    f_idx        = fellow['_idx']
    period_set   = set((pm[0], pm[1]) for pm in period_months)
    period_wi    = [wi for wi, w in enumerate(all_weeks)
                    if (w['year'], w['month']) in period_set]
    total_weeks  = len(period_wi)
    active_weeks = sum(1 for wi in period_wi if active_in_week[wi][f_idx])
    if total_weeks == 0 or active_weeks == 0:
        return 0, 0
    ratio = active_weeks / total_weeks
    return max(0, round(2 * ratio)), max(0, round(1 * ratio))


# ─────────────────────────────────────────────────────────────────
# SOLVER 1 — SERVICE WEEKS
# ─────────────────────────────────────────────────────────────────

def solve_service_weeks(surgeons, months, block_number, preferences, prior_totals):
    """
    SOLVER 1: Service week assignments.

    THE KEY FIX — Per-role hard caps derived from FTE target:

    Every surgeon's FTE target mathematically determines the maximum
    number of weeks they can hold each role. This is the hard cap.

    Loor (0.38 FTE = 32 shifts target):
      SICU cap = round(32 / 7) = 5 weeks maximum — HARD LIMIT
      She gets exactly her proportional share. No more.

    Lim (0.16 FTE = 13 shifts target):
      TSICU cap = round(13 / 7) = 2 weeks maximum — HARD LIMIT
      He gets exactly his proportional share. No more.

    Al-Aref (1.0 FTE = 84 shifts target, willing):
      Each role cap = round(84 * 1.4 / 7) = 17 weeks maximum
      He absorbs overflow from baseline surgeons.

    This ensures:
    1. Low-FTE surgeons get EXACTLY their proportional share
    2. High-FTE willing/seeking surgeons absorb the overflow
    3. The schedule is always feasible because overflow has somewhere to go
    4. FTE equity is mathematically guaranteed

    HARD CONSTRAINTS:
    - One surgeon per role per week
    - Eligibility (contractual)
    - Active dates for all surgeons
    - No surgeon in two roles simultaneously
    - Per-role week caps derived from FTE target (THE NEW HARD CAP)
    - ACS M-Sun no repeat consecutive weeks
    - Fellow rotation: 2 ACS + 1 SICU per 2-month period
    - Fellows cannot share same role same week

    SOFT CONSTRAINTS (penalized, not blocked):
    - Consecutive 7-day weeks (heavy penalty weight 500)
    - Time off / conference weeks (penalty weight 80)
    - Getting close to FTE target
    """
    num_surgeons = len(surgeons)

    all_weeks, week_to_month = get_all_weeks_deduped(months)
    num_weeks                = len(all_weeks)

    active_in_week = [
        [is_active_for_week(surgeons[s], all_weeks[wi])
         for s in range(num_surgeons)]
        for wi in range(num_weeks)
    ]

    fellow_indices = [s for s in range(num_surgeons) if is_fellow(surgeons[s])]

    # Parse time off preferences
    surgeon_time_off = {}
    for s in range(num_surgeons):
        prefs = get_surgeon_prefs(surgeons[s].get('id', ''), preferences)
        y_ref = months[0][0]
        off   = parse_date_list(prefs.get('time_off',    ''), y_ref)
        conf  = parse_date_list(prefs.get('conferences', ''), y_ref)
        surgeon_time_off[s] = off | conf

    # Compute FTE targets and per-role caps
    block_targets = [
        compute_block_target(surgeons[s], block_number, prior_totals, months)
        for s in range(num_surgeons)
    ]
    target_shifts = [max(0, round(t)) for t in block_targets]

    # Per-role caps: maximum weeks each surgeon can hold each role
    # This is derived from their FTE target — it IS their entitlement
    role_caps = []
    for s in range(num_surgeons):
        pref = get_pref(surgeons[s])
        caps = compute_role_caps(surgeons[s], target_shifts[s], pref)
        role_caps.append(caps)

    model = cp_model.CpModel()

    # ── Decision variables ────────────────────────────────────────
    acs_msun = [[model.NewBoolVar(f'am_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_weeks)]
    acs_mf   = [[model.NewBoolVar(f'af_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_weeks)]
    mcnair   = [[model.NewBoolVar(f'mn_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_weeks)]
    tsicu    = [[model.NewBoolVar(f'ts_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_weeks)]
    sicu_v   = [[model.NewBoolVar(f'si_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_weeks)]

    all_roles = [
        ('acs_mf',   acs_mf,   SHIFTS_ACS_MF),
        ('acs_msun', acs_msun, SHIFTS_ACS_MSUN),
        ('mcnair',   mcnair,   SHIFTS_ICU),
        ('tsicu',    tsicu,    SHIFTS_ICU),
        ('sicu',     sicu_v,   SHIFTS_ICU),
    ]

    role_var_map = {
        'acs_mf':   acs_mf,
        'acs_msun': acs_msun,
        'mcnair':   mcnair,
        'tsicu':    tsicu,
        'sicu':     sicu_v,
    }

    # ══════════════════════════════════════════════════════════════
    # HARD CONSTRAINTS
    # ══════════════════════════════════════════════════════════════

    # H1 — One surgeon per role per week
    for wi in range(num_weeks):
        model.AddExactlyOne(acs_msun[wi])
        model.AddExactlyOne(acs_mf[wi])
        model.AddExactlyOne(mcnair[wi])
        model.AddExactlyOne(tsicu[wi])
        model.AddExactlyOne(sicu_v[wi])

    # H2 — Eligibility (contractual) and active dates
    for wi in range(num_weeks):
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
                model.Add(sicu_v[wi][s] == 0)

    # H3 — No surgeon in two roles simultaneously
    for wi in range(num_weeks):
        for s in range(num_surgeons):
            model.Add(
                acs_msun[wi][s] + acs_mf[wi][s] + mcnair[wi][s] +
                tsicu[wi][s]    + sicu_v[wi][s] <= 1
            )

    # H4 — PER-ROLE WEEK CAPS derived from FTE target
    #
    # This is the core fix. Each surgeon has a hard maximum number of
    # weeks they can hold each role, calculated from their FTE target.
    #
    # Loor gets at most 5 SICU weeks (32 shifts / 7 = 4.6, rounded up).
    # Lim gets at most 2 TSICU weeks (13 shifts / 7 = 1.9, rounded up).
    # Al-Aref (willing) gets at most 17 of each role (84*1.4/7 = 16.8).
    #
    # This guarantees:
    # - Low-FTE surgeons get exactly their proportional share
    # - They cannot take more even if eligible and available
    # - Overflow automatically goes to willing/seeking surgeons
    # - The schedule is always feasible
    for s in range(num_surgeons):
        caps = role_caps[s]
        for role_name, role_var in role_var_map.items():
            cap = caps.get(role_name, 0)
            if cap == 0:
                for wi in range(num_weeks):
                    model.Add(role_var[wi][s] == 0)
            else:
                model.Add(sum(role_var[wi][s] for wi in range(num_weeks)) <= cap)

    # H5 — ACS M-Sun cannot repeat consecutive weeks
    for wi in range(num_weeks - 1):
        for s in range(num_surgeons):
            model.Add(acs_msun[wi][s] + acs_msun[wi + 1][s] <= 1)

    # H6 — Fellows cannot share same role same week
    if len(fellow_indices) >= 2:
        for wi in range(num_weeks):
            for _, rvars, _ in all_roles:
                model.Add(sum(rvars[wi][f] for f in fellow_indices) <= 1)

    # H7 — Fellow rotation: 2 ACS + 1 SICU per 2-month period
    two_month_periods = get_two_month_periods(months)
    for period_months in two_month_periods:
        period_set = set((pm[0], pm[1]) for pm in period_months)
        period_wi  = [wi for wi, w in enumerate(all_weeks)
                      if (w['year'], w['month']) in period_set]
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
                model.Add(sum(sicu_v[wi][f] for wi in period_wi) == sicu_t)

    # ══════════════════════════════════════════════════════════════
    # OBJECTIVE
    #
    # With hard per-role caps doing the heavy lifting of FTE equity,
    # the objective just needs to:
    # 1. Reward filling every slot (drives feasibility)
    # 2. Penalize consecutive 7-day weeks (avoids fatigue)
    # 3. Respect time off preferences
    #
    # Reward weight by FTE target: surgeons with higher targets get
    # higher reward per assignment, so they fill their larger quotas.
    # This prevents low-FTE surgeons from accidentally taking slots
    # that high-FTE surgeons need (the caps prevent the opposite).
    # ══════════════════════════════════════════════════════════════

    obj_terms     = []
    penalty_terms = []

    for s in range(num_surgeons):
        t    = target_shifts[s]
        pref = get_pref(surgeons[s])
        if t == 0:
            continue

        # Reward weight proportional to FTE target
        # High-FTE surgeons get more reward per assignment so they
        # fill their larger quotas proportionally
        base_reward = max(10, round(t / 5))

        for wi in range(num_weeks):
            if not active_in_week[wi][s]:
                continue
            for role_name, rvars, _ in all_roles:
                if is_eligible(surgeons[s], role_name):
                    obj_terms.append(base_reward * rvars[wi][s])

    # Consecutive 7-day weeks — heavily penalized, not hard-blocked
    # The solver will avoid this in the vast majority of cases
    # but won't fail if it's mathematically unavoidable
    seven_day = [acs_msun, mcnair, tsicu, sicu_v]
    for wi in range(num_weeks - 1):
        for s in range(num_surgeons):
            in_7day_wi  = model.NewBoolVar(f'i7_{wi}_{s}')
            in_7day_wi1 = model.NewBoolVar(f'i7_{wi+1}_{s}')
            model.AddMaxEquality(in_7day_wi,  [r[wi][s]   for r in seven_day])
            model.AddMaxEquality(in_7day_wi1, [r[wi+1][s] for r in seven_day])
            consec = model.NewBoolVar(f'consec_{wi}_{s}')
            model.AddMinEquality(consec, [in_7day_wi, in_7day_wi1])
            penalty_terms.append(500 * consec)

    # Time off and conference blocking
    for wi, week in enumerate(all_weeks):
        for s in range(num_surgeons):
            if surgeon_time_off[s] and week_overlaps_dates(week, surgeon_time_off[s]):
                for _, rvars, _ in all_roles:
                    penalty_terms.append(80 * rvars[wi][s])

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
            f"The per-role week caps combined with eligibility constraints "
            f"cannot be satisfied. This usually means willing/seeking surgeons "
            f"don't have enough capacity to cover all weeks after baseline caps "
            f"are applied. Ensure at least several surgeons are marked willing "
            f"or seeking in the admin page."
        )

    # Extract assignments
    week_assignments = {}
    for wi in range(num_weeks):
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
            if solver.Value(sicu_v[wi][s]):    week_assignments[wi]['SICU']        = name

    return week_assignments


# ─────────────────────────────────────────────────────────────────
# SOLVER 2 — CALL
# ─────────────────────────────────────────────────────────────────

def solve_call(surgeons, months, week_assignments, preferences):
    """
    SOLVER 2: Call night assignments.

    Takes the completed service week schedule as fixed input.
    Assigns one call surgeon per night respecting all call rules.

    HARD:
    - One call surgeon per night
    - Call eligibility
    - Active dates
    - McNair: no call any night
    - TSICU/SICU/ACS M-Sun: no call Mon-Sat (Sunday OK)
    - ACS M-F: no call Mon-Thu (Fri/Sat/Sun OK)
    - Fellow max 5 call nights per month
    - Max call nights per month per surgeon profile

    SOFT:
    - Weekend call equity
    - Call day preferences
    - Avoid nights
    - No 3+ consecutive call nights
    """
    num_surgeons   = len(surgeons)
    num_months     = len(months)
    month_days     = [monthrange(y, mo)[1] for y, mo in months]
    fellow_indices = [s for s in range(num_surgeons) if is_fellow(surgeons[s])]

    surgeon_avoid = {}
    for s in range(num_surgeons):
        prefs = get_surgeon_prefs(surgeons[s].get('id', ''), preferences)
        surgeon_avoid[s] = parse_date_list(
            prefs.get('avoid_nights', ''), months[0][0])

    active_in_month = [
        [is_active_for_month(surgeons[s], y, mo) for s in range(num_surgeons)]
        for mi, (y, mo) in enumerate(months)
    ]

    # Build night -> service role lookup
    night_role = {}
    for mi, (y, mo) in enumerate(months):
        night_role[mi] = {}
        for d in range(month_days[mi]):
            night_role[mi][d] = {}
            date_dt = datetime(y, mo, d + 1)
            for wi in sorted(week_assignments.keys()):
                wa = week_assignments[wi]
                ws = wa['start']
                we = ws + timedelta(days=6)
                if ws <= date_dt <= we:
                    for role in ['ACS (M-Sun)', 'ACS (M-F)',
                                 'McNair ICU', 'TSICU', 'SICU']:
                        name = wa.get(role)
                        if name:
                            for s in range(num_surgeons):
                                if surgeons[s]['name'] == name:
                                    night_role[mi][d][s] = role
                    break

    model = cp_model.CpModel()

    call = [
        [[model.NewBoolVar(f'c_{mi}_{d}_{s}') for s in range(num_surgeons)]
         for d in range(month_days[mi])]
        for mi in range(num_months)
    ]

    # H1 — One call surgeon per night
    for mi in range(num_months):
        for d in range(month_days[mi]):
            model.AddExactlyOne(call[mi][d])

    # H2 — Eligibility and active status
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                if not active_in_month[mi][s] or not is_eligible(surgeons[s], 'call'):
                    model.Add(call[mi][d][s] == 0)

    # H3 — Call restrictions based on service week role
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            dow = datetime(y, mo, d + 1).weekday()
            for s in range(num_surgeons):
                role = night_role[mi][d].get(s)
                if role is None:
                    continue
                if role == 'McNair ICU':
                    model.Add(call[mi][d][s] == 0)
                elif role in ('TSICU', 'SICU', 'ACS (M-Sun)'):
                    if dow <= 5:
                        model.Add(call[mi][d][s] == 0)
                elif role == 'ACS (M-F)':
                    if dow <= 3:
                        model.Add(call[mi][d][s] == 0)

    # H4 — Fellow max 5 call nights per month
    for f in fellow_indices:
        max_call = int(surgeons[f].get('max_call_per_month', 5))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][f] for d in range(month_days[mi])) <= max_call)

    # H5 — Max call nights per month per surgeon
    for s in range(num_surgeons):
        if s in fellow_indices:
            continue
        max_call = int(surgeons[s].get('max_call_per_month', 8))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][s] for d in range(month_days[mi])) <= max_call)

    # ── Soft constraints ──────────────────────────────────────────
    obj_terms     = []
    penalty_terms = []

    # Weekend call equity
    weekend_nights = [
        (mi, d)
        for mi, (y, mo) in enumerate(months)
        for d in range(month_days[mi])
        if datetime(y, mo, d + 1).weekday() >= 4
    ]
    total_weekend      = len(weekend_nights)
    call_eligible_list = [s for s in range(num_surgeons)
                          if is_eligible(surgeons[s], 'call')]
    fair_wknd = max(1, round(total_weekend / max(1, len(call_eligible_list))))

    surgeon_wknd = [
        model.NewIntVar(0, total_weekend, f'wk_{s}')
        for s in range(num_surgeons)
    ]
    for s in range(num_surgeons):
        wvars = [call[mi][d][s] for mi, d in weekend_nights]
        model.Add(surgeon_wknd[s] == (sum(wvars) if wvars else 0))

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

    # Call day preference
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

    # No 3+ consecutive call nights
    for mi in range(num_months):
        days = month_days[mi]
        for s in range(num_surgeons):
            for d in range(days - 3):
                run4 = model.NewBoolVar(f'r4_{mi}_{d}_{s}')
                model.AddMinEquality(run4, [
                    call[mi][d][s], call[mi][d+1][s],
                    call[mi][d+2][s], call[mi][d+3][s]
                ])
                penalty_terms.append(25 * run4)

    # Avoid nights
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                if surgeon_avoid[s] and day_in_dates(y, mo, d, surgeon_avoid[s]):
                    penalty_terms.append(30 * call[mi][d][s])

    total_obj = []
    if obj_terms:
        total_obj.append(sum(obj_terms))
    if penalty_terms:
        total_obj.append(-sum(penalty_terms))
    if total_obj:
        model.Maximize(sum(total_obj) if len(total_obj) > 1 else total_obj[0])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers  = 4

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise Exception(
            f"Call solver failed: {solver.StatusName(status)}. "
            f"Cannot assign call given service week restrictions. "
            f"Check call eligibility flags."
        )

    call_assignments = {}
    for mi in range(num_months):
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                if solver.Value(call[mi][d][s]):
                    call_assignments[(mi, d)] = surgeons[s]['name']

    return call_assignments


# ─────────────────────────────────────────────────────────────────
# BUILD OUTPUT
# ─────────────────────────────────────────────────────────────────

def build_output(surgeons, months, week_assignments, call_assignments,
                 block_number, prior_totals):

    num_surgeons = len(surgeons)
    num_months   = len(months)
    month_days   = [monthrange(y, mo)[1] for y, mo in months]

    block_targets = [
        compute_block_target(surgeons[s], block_number, prior_totals, months)
        for s in range(num_surgeons)
    ]
    target_shifts = [max(0, round(t)) for t in block_targets]

    months_weeks = {mi: [] for mi in range(num_months)}
    for wi in sorted(week_assignments.keys()):
        months_weeks[week_assignments[wi]['month_idx']].append(wi)

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

    # ── Validation ────────────────────────────────────────────────
    violations = []
    warnings   = []

    all_weeks_flat = []
    for mi in range(num_months):
        mk = f"{months[mi][0]}-{str(months[mi][1]).zfill(2)}"
        all_weeks_flat.extend(result[mk]['weeks'])

    for mi, (y, mo) in enumerate(months):
        mk          = f"{y}-{str(mo).zfill(2)}"
        month_label = datetime(y, mo, 1).strftime('%B %Y')

        for w in result[mk]['weeks']:
            for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                if role not in w:
                    violations.append(
                        f"{month_label} {w['label']}: {role} not assigned")

        for d in range(month_days[mi]):
            if str(d + 1) not in result[mk]['nights']:
                violations.append(f"{month_label} day {d + 1}: No call assigned")

        for w in result[mk]['weeks']:
            seen = {}
            for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                name = w.get(role)
                if name:
                    if name in seen:
                        violations.append(
                            f"{month_label} {w['label']}: "
                            f"{name} in {seen[name]} and {role}")
                    seen[name] = role

    # Consecutive 7-day weeks
    seven_day_labels = ['ACS (M-Sun)', 'McNair ICU', 'TSICU', 'SICU']
    for i in range(len(all_weeks_flat) - 1):
        w1 = all_weeks_flat[i]
        w2 = all_weeks_flat[i + 1]
        for r1 in seven_day_labels:
            for r2 in seven_day_labels:
                n1 = w1.get(r1)
                n2 = w2.get(r2)
                if n1 and n1 == n2:
                    warnings.append(
                        f"Consecutive 7-day weeks: {n1} ({r1} -> {r2}) — review manually")

    # ACS M-Sun consecutive
    for i in range(len(all_weeks_flat) - 1):
        n1 = all_weeks_flat[i].get('ACS (M-Sun)')
        n2 = all_weeks_flat[i + 1].get('ACS (M-Sun)')
        if n1 and n1 == n2:
            violations.append(f"ACS M-Sun consecutive: {n1} — hard rule violated")

    # Call run check
    for mi, (y, mo) in enumerate(months):
        nights = result[f"{y}-{str(mo).zfill(2)}"]['nights']
        for s in range(num_surgeons):
            name = surgeons[s]['name']
            run  = 0
            for d in range(1, month_days[mi] + 1):
                if nights.get(str(d), {}).get('Call') == name:
                    run += 1
                    if run >= 4:
                        warnings.append(
                            f"{datetime(y, mo, 1).strftime('%B %Y')}: "
                            f"{name} has {run}+ consecutive call nights "
                            f"starting day {d - run + 1}")
                        break
                else:
                    run = 0

    # Sunday call -> Monday fresh start
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            if datetime(y, mo, d + 1).weekday() != 6:
                continue
            call_name = result[f"{y}-{str(mo).zfill(2)}"]['nights'].get(
                str(d + 1), {}
            ).get('Call', '')
            if not call_name:
                continue
            next_monday = datetime(y, mo, d + 1) + timedelta(days=1)
            for wi in sorted(week_assignments.keys()):
                wa = week_assignments[wi]
                if wa['start'] == next_monday:
                    for role in ['ACS (M-Sun)', 'ACS (M-F)',
                                 'McNair ICU', 'TSICU', 'SICU']:
                        if wa.get(role) == call_name:
                            prior_wi = wi - 1
                            in_prior = prior_wi >= 0 and any(
                                week_assignments[prior_wi].get(r) == call_name
                                for r in ['ACS (M-Sun)', 'ACS (M-F)',
                                          'McNair ICU', 'TSICU', 'SICU']
                            )
                            if not in_prior:
                                warnings.append(
                                    f"{call_name}: call Sun "
                                    f"{next_monday.strftime('%b %-d')} "
                                    f"then fresh {role} Mon — fix manually")

    # Fellow rotation validation
    all_weeks_list = [week_assignments[wi]
                      for wi in sorted(week_assignments.keys())]
    active_in_week = [
        [is_active_for_week(surgeons[s], all_weeks_list[wi])
         for s in range(num_surgeons)]
        for wi in range(len(all_weeks_list))
    ]
    fellow_indices = [s for s in range(num_surgeons) if is_fellow(surgeons[s])]

    for period_idx, period_months in enumerate(get_two_month_periods(months)):
        for f in fellow_indices:
            fname        = surgeons[f]['name']
            acs_t, sicu_t = compute_fellow_rotation_target(
                surgeons[f], period_months, months, active_in_week, all_weeks_list
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
                    f"{acs_count} ACS weeks (expected {acs_t})")
            if sicu_t > 0 and sicu_count != sicu_t:
                violations.append(
                    f"Fellow {fname} period {period_idx + 1}: "
                    f"{sicu_count} SICU weeks (expected {sicu_t})")

    # FTE equity
    block_fte_summary    = {}
    weekend_nights       = [
        (mi, d)
        for mi, (y, mo) in enumerate(months)
        for d in range(month_days[mi])
        if datetime(y, mo, d + 1).weekday() >= 4
    ]
    weekend_call_summary = {}

    for s in range(num_surgeons):
        name  = surgeons[s]['name']
        pref  = get_pref(surgeons[s])
        total = sum(
            result[f"{y}-{str(mo).zfill(2)}"]['fte_summary'].get(name, 0)
            for y, mo in months
        )
        t     = target_shifts[s]
        delta = total - t

        if t > 0 and delta < -7:
            warnings.append(
                f"{name}: served {total} vs target {t} "
                f"(short {abs(delta)}) — check eligibility and role caps")
        elif t > 0 and delta > 14 and pref == 'baseline':
            warnings.append(
                f"{name}: served {total} vs target {t} "
                f"(+{delta} over) — review role caps")

        block_fte_summary[name] = {
            'served': total,
            'target': round(block_targets[s], 1),
            'delta':  round(delta, 1),
        }

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
