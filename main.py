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
    return jsonify({'status': 'ok', 'service': 'ORbit Solver v13'})


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

        # Two solvers: service weeks first, then call
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
    """How many of the 5 service roles is this surgeon eligible for."""
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
    Block target = 84 x FTE for Block 1.
    Block 2 = (168 x FTE) minus Block 1 actuals.
    Prorated for start/departure dates.

    This is the equity baseline — every surgeon gets their
    proportional share of work. The FTE system ensures fairness.
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
    from datetime import date
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


def compute_fellow_rotation_target(fellow, period_months, months, active_in_week, all_weeks):
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
    Assigns the 5 weekly service roles across the full 6-month block.

    PHILOSOPHY:
    Think like a human scheduler who has been doing this for 20 years.

    Step 1: Honor contractual obligations — eligibility flags are hard.
            A surgeon can only be assigned to roles in their contract.

    Step 2: Fill single-role surgeons first — they have no flexibility.
            Loor (SICU only), Chatterjee (TSICU only), Perez (ACS only),
            Bonville/Lim (TSICU only), Rojas-Khalil (SICU only).
            Give them their proportional FTE share and stop.

    Step 3: Fill remaining slots with multi-role surgeons proportionally.
            Al-Aref, Dumas, Fitzgerald etc absorb whatever is left,
            weighted by their FTE targets.

    Step 4: Enforce fellow rotation requirements.

    HARD CONSTRAINTS (never violated):
    - Exactly one surgeon per role per week
    - Eligibility (contractual — never overridden)
    - Active dates for all surgeons
    - No surgeon in two roles simultaneously
    - Fellow rotation: 2 ACS + 1 SICU per 2-month period
    - Fellows cannot share same role same week
    - ACS M-Sun cannot repeat consecutive weeks (contractual)

    SOFT CONSTRAINTS (strongly preferred, flagged if violated):
    - No consecutive 7-day weeks — heavily penalized but not hard-blocked
      because in rare situations it may be mathematically unavoidable
      given the pool size and FTE constraints
    - FTE equity — each surgeon gets close to their block target
      Baseline: target is a ceiling, over-assignment heavily penalized
      Willing/seeking: can absorb overflow beyond their target
    - Time off / conference weeks
    - Holiday preferences
    """
    num_surgeons = len(surgeons)

    all_weeks, week_to_month = get_all_weeks_deduped(months)
    num_weeks                = len(all_weeks)

    active_in_week = [
        [is_active_for_week(surgeons[s], all_weeks[wi]) for s in range(num_surgeons)]
        for wi in range(num_weeks)
    ]

    fellow_indices = [s for s in range(num_surgeons) if is_fellow(surgeons[s])]

    # Parse preferences
    surgeon_time_off = {}
    for s in range(num_surgeons):
        prefs = get_surgeon_prefs(surgeons[s].get('id', ''), preferences)
        y_ref = months[0][0]
        off   = parse_date_list(prefs.get('time_off',    ''), y_ref)
        conf  = parse_date_list(prefs.get('conferences', ''), y_ref)
        surgeon_time_off[s] = off | conf

    # FTE targets
    block_targets = [
        compute_block_target(surgeons[s], block_number, prior_totals, months)
        for s in range(num_surgeons)
    ]
    target_shifts = [max(0, round(t)) for t in block_targets]

    # Classify surgeons by role flexibility
    # single_role: only eligible for 1 service role — fill first
    # multi_role:  eligible for 2+ service roles — fill remaining slots
    single_role = [s for s in range(num_surgeons)
                   if count_service_roles(surgeons[s]) == 1
                   and not is_fellow(surgeons[s])]
    multi_role  = [s for s in range(num_surgeons)
                   if count_service_roles(surgeons[s]) > 1
                   and not is_fellow(surgeons[s])]

    model = cp_model.CpModel()

    # ── Variables ─────────────────────────────────────────────────
    acs_msun = [[model.NewBoolVar(f'am_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_weeks)]
    acs_mf   = [[model.NewBoolVar(f'af_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_weeks)]
    mcnair   = [[model.NewBoolVar(f'mn_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_weeks)]
    tsicu    = [[model.NewBoolVar(f'ts_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_weeks)]
    sicu     = [[model.NewBoolVar(f'si_{wi}_{s}') for s in range(num_surgeons)]
                for wi in range(num_weeks)]

    all_roles = [
        ('acs_mf',   acs_mf,   SHIFTS_ACS_MF),
        ('acs_msun', acs_msun, SHIFTS_ACS_MSUN),
        ('mcnair',   mcnair,   SHIFTS_ICU),
        ('tsicu',    tsicu,    SHIFTS_ICU),
        ('sicu',     sicu,     SHIFTS_ICU),
    ]

    # Per-surgeon total shifts
    surgeon_total = [model.NewIntVar(0, 250, f'tot_{s}') for s in range(num_surgeons)]
    for s in range(num_surgeons):
        terms = []
        for wi in range(num_weeks):
            if not active_in_week[wi][s]:
                continue
            terms += [
                SHIFTS_ACS_MF   * acs_mf[wi][s],
                SHIFTS_ACS_MSUN * acs_msun[wi][s],
                SHIFTS_ICU       * mcnair[wi][s],
                SHIFTS_ICU       * tsicu[wi][s],
                SHIFTS_ICU       * sicu[wi][s],
            ]
        model.Add(surgeon_total[s] == (sum(terms) if terms else 0))

    # ══════════════════════════════════════════════════════════════
    # HARD CONSTRAINTS — these are never violated
    # ══════════════════════════════════════════════════════════════

    # H1 — Exactly one surgeon per role per week
    for wi in range(num_weeks):
        model.AddExactlyOne(acs_msun[wi])
        model.AddExactlyOne(acs_mf[wi])
        model.AddExactlyOne(mcnair[wi])
        model.AddExactlyOne(tsicu[wi])
        model.AddExactlyOne(sicu[wi])

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
                model.Add(sicu[wi][s] == 0)

    # H3 — No surgeon in two roles simultaneously
    for wi in range(num_weeks):
        for s in range(num_surgeons):
            model.Add(
                acs_msun[wi][s] + acs_mf[wi][s] + mcnair[wi][s] +
                tsicu[wi][s]    + sicu[wi][s] <= 1
            )

    # H4 — ACS M-Sun cannot repeat consecutive weeks (contractual rule)
    for wi in range(num_weeks - 1):
        for s in range(num_surgeons):
            model.Add(acs_msun[wi][s] + acs_msun[wi + 1][s] <= 1)

    # H5 — Fellows cannot share same role same week
    if len(fellow_indices) >= 2:
        for wi in range(num_weeks):
            for _, rvars, _ in all_roles:
                model.Add(sum(rvars[wi][f] for f in fellow_indices) <= 1)

    # H6 — Fellow rotation: exactly 2 ACS + 1 SICU per 2-month period
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
                model.Add(sum(sicu[wi][f] for wi in period_wi) == sicu_t)

    # ══════════════════════════════════════════════════════════════
    # OBJECTIVE — human-like priority scheduling
    #
    # The objective encodes the human scheduler's decision process:
    #
    # PRIORITY 1 (weight 1000): Fill single-role surgeons to their target.
    # These surgeons (Loor, Chatterjee, Perez, Bonville, Lim, Rojas-Khalil)
    # have no flexibility — give them their proportional FTE share first.
    # The 1000-weight reward means the solver fills them before anyone else.
    # The 2000-weight penalty for going over means it stops at their target.
    #
    # PRIORITY 2 (weight 400): Fill fellows per rotation requirements.
    # H6 hard constraint handles the rotation structure. The weight here
    # drives proportional filling within their allowed weeks.
    #
    # PRIORITY 3 (weight 100): Fill multi-role surgeons with remainder.
    # Al-Aref, Dumas, Fitzgerald etc absorb whatever weeks remain.
    # They are proportionally weighted by FTE target — a 1.0 FTE surgeon
    # gets more weeks than a 0.5 FTE surgeon from the same pool.
    # Willing/seeking surgeons get additional overflow beyond their target.
    #
    # PENALTY — consecutive 7-day weeks (weight 500):
    # Very heavily penalized so the solver strongly avoids it.
    # Not a hard constraint because in rare edge cases (very small active
    # pool for a specific role in a specific month) it may be unavoidable.
    # The validation report flags any that occur for manual review.
    #
    # PENALTY — over-target (weight 300 baseline, 30 willing, 5 seeking):
    # Baseline surgeons are strongly discouraged from going over their target.
    # Willing/seeking absorb overflow naturally.
    # ══════════════════════════════════════════════════════════════

    obj_terms     = []
    penalty_terms = []

    for s in range(num_surgeons):
        t    = target_shifts[s]
        pref = get_pref(surgeons[s])

        if t == 0:
            continue

        # Determine priority weight based on surgeon type
        if s in single_role:
            reward       = 1000  # Fill these first
            over_penalty = 2000  # Stop at their target
        elif s in fellow_indices:
            reward       = 400
            over_penalty = 800
        else:
            # Multi-role surgeons — weight proportional to FTE target
            # so 1.0 FTE surgeon gets more weeks than 0.5 FTE surgeon
            reward = max(50, round(100 * t / 84))
            if pref == 'baseline':
                over_penalty = 300
            elif pref == 'willing':
                over_penalty = 30
            else:  # seeking
                over_penalty = 5

        # Reward assignments up to target, penalize beyond
        # We do this per-week per-role using running counts
        # to avoid needing complex integer tracking variables
        role_week_count = {rn: 0 for rn, _, _ in all_roles}

        # Weeks to fill per role to hit target
        weeks_cap = {
            'acs_mf':   max(1, round(t / SHIFTS_ACS_MF)),
            'acs_msun': max(1, round(t / SHIFTS_ACS_MSUN)),
            'mcnair':   max(1, round(t / SHIFTS_ICU)),
            'tsicu':    max(1, round(t / SHIFTS_ICU)),
            'sicu':     max(1, round(t / SHIFTS_ICU)),
        }

        for wi in range(num_weeks):
            if not active_in_week[wi][s]:
                continue
            for rn, rvars, _ in all_roles:
                if not is_eligible(surgeons[s], rn):
                    continue
                wc  = role_week_count[rn]
                cap = weeks_cap[rn]
                if wc < cap:
                    obj_terms.append(reward * rvars[wi][s])
                else:
                    penalty_terms.append(over_penalty * rvars[wi][s])
                role_week_count[rn] += 1

    # Consecutive 7-day week penalty — heavily discouraged but not hard-blocked
    # A human scheduler avoids this but will do it in rare unavoidable situations
    seven_day = [acs_msun, mcnair, tsicu, sicu]
    for wi in range(num_weeks - 1):
        for s in range(num_surgeons):
            # Use AddBoolAnd to detect back-to-back 7-day assignments
            for r1 in seven_day:
                for r2 in seven_day:
                    # If surgeon S is in a 7-day role both this week AND next week
                    # penalize heavily. We approximate using individual role terms
                    # since multiplying booleans requires auxiliary vars.
                    # Simple approach: penalize each 7-day assignment with small weight,
                    # net effect is double-penalty for consecutive weeks.
                    pass

            # Cleaner approach: aux var for "in any 7-day role this week"
            in_7day_wi = model.NewBoolVar(f'i7_{wi}_{s}')
            model.AddMaxEquality(in_7day_wi, [r[wi][s] for r in seven_day])
            in_7day_wi1 = model.NewBoolVar(f'i7_{wi+1}_{s}')
            model.AddMaxEquality(in_7day_wi1, [r[wi+1][s] for r in seven_day])
            consec = model.NewBoolVar(f'consec_{wi}_{s}')
            model.AddMinEquality(consec, [in_7day_wi, in_7day_wi1])
            penalty_terms.append(500 * consec)

    # Time off and conference soft blocking
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
            f"Hard constraints (eligibility, active dates, fellow rotation, "
            f"ACS M-Sun no-repeat) cannot all be satisfied simultaneously. "
            f"Check surgeon eligibility flags in the admin page."
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
            if solver.Value(sicu[wi][s]):      week_assignments[wi]['SICU']        = name

    return week_assignments


# ─────────────────────────────────────────────────────────────────
# SOLVER 2 — CALL
# ─────────────────────────────────────────────────────────────────

def solve_call(surgeons, months, week_assignments, preferences):
    """
    Assigns one call surgeon per night using the completed
    service week schedule as fixed input.

    HARD CONSTRAINTS:
    - Exactly one call surgeon per night
    - Call eligibility respected
    - Active dates respected
    - McNair surgeon: no call any night that week
    - TSICU/SICU/ACS M-Sun: no call Mon-Sat (Sunday OK)
    - ACS M-F: no call Mon-Thu (Fri/Sat/Sun OK)
    - Fellow max 5 call nights per month

    SOFT CONSTRAINTS:
    - Weekend call equity — fair share per eligible surgeon
    - Call day preferences (e.g. Rojas-Khalil Fri/Sat)
    - Avoid nights from preferences
    - No 3+ consecutive call nights
    """
    num_surgeons  = len(surgeons)
    num_months    = len(months)
    month_days    = [monthrange(y, mo)[1] for y, mo in months]
    fellow_indices = [s for s in range(num_surgeons) if is_fellow(surgeons[s])]

    surgeon_avoid = {}
    for s in range(num_surgeons):
        prefs = get_surgeon_prefs(surgeons[s].get('id', ''), preferences)
        surgeon_avoid[s] = parse_date_list(prefs.get('avoid_nights', ''), months[0][0])

    active_in_month = [
        [is_active_for_month(surgeons[s], y, mo) for s in range(num_surgeons)]
        for mi, (y, mo) in enumerate(months)
    ]

    # Build night -> service role lookup from week_assignments
    # night_role[mi][d][s] = role name if surgeon s is on service that night
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
                    for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
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
                    if dow <= 5:  # Mon-Sat blocked, Sunday OK
                        model.Add(call[mi][d][s] == 0)
                elif role == 'ACS (M-F)':
                    if dow <= 3:  # Mon-Thu blocked, Fri/Sat/Sun OK
                        model.Add(call[mi][d][s] == 0)

    # H4 — Fellow max 5 call nights per month
    for f in fellow_indices:
        max_call = int(surgeons[f].get('max_call_per_month', 5))
        for mi in range(num_months):
            model.Add(sum(call[mi][d][f] for d in range(month_days[mi])) <= max_call)

    # H5 — Max call nights per month per surgeon (from profile)
    for s in range(num_surgeons):
        if s in fellow_indices:
            continue
        max_call = int(surgeons[s].get('max_call_per_month', 8))
        for mi in range(num_months):
            model.Add(sum(call[mi][d][s] for d in range(month_days[mi])) <= max_call)

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

    surgeon_wknd = [model.NewIntVar(0, total_weekend, f'wk_{s}')
                    for s in range(num_surgeons)]
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
                model.AddMinEquality(run4, [call[mi][d][s], call[mi][d+1][s],
                                            call[mi][d+2][s], call[mi][d+3][s]])
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
            f"Cannot assign call to all nights given service week restrictions. "
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

    # Group weeks by month
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

        for w in result[mk]['weeks']:
            for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                if role not in w:
                    violations.append(f"{month_label} {w['label']}: {role} not assigned")

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
                            f"{month_label} {w['label']}: {name} in {seen[name]} and {role}"
                        )
                    seen[name] = role

    # Consecutive 7-day weeks (flagged — solver strongly avoids but may allow in edge cases)
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
                        f"Consecutive 7-day weeks: {n1} ({r1} -> {r2}) — review manually"
                    )

    # ACS M-Sun consecutive (hard rule — should never appear)
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
                            f"{name} has {run}+ consecutive call nights starting day {d - run + 1}"
                        )
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
                    for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                        if wa.get(role) == call_name:
                            prior_wi = wi - 1
                            in_prior = prior_wi >= 0 and any(
                                week_assignments[prior_wi].get(r) == call_name
                                for r in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']
                            )
                            if not in_prior:
                                warnings.append(
                                    f"{call_name}: call Sun {next_monday.strftime('%b %-d')} "
                                    f"then fresh {role} Mon — fix manually"
                                )

    # Fellow rotation validation
    all_weeks_list = [week_assignments[wi] for wi in sorted(week_assignments.keys())]
    active_in_week = [
        [is_active_for_week(surgeons[s], all_weeks_list[wi])
         for s in range(num_surgeons)]
        for wi in range(len(all_weeks_list))
    ]
    fellow_indices = [s for s in range(num_surgeons) if is_fellow(surgeons[s])]

    for period_idx, period_months in enumerate(get_two_month_periods(months)):
        for f in fellow_indices:
            fname       = surgeons[f]['name']
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
                    f"{acs_count} ACS weeks (expected {acs_t})"
                )
            if sicu_t > 0 and sicu_count != sicu_t:
                violations.append(
                    f"Fellow {fname} period {period_idx + 1}: "
                    f"{sicu_count} SICU weeks (expected {sicu_t})"
                )

    # FTE equity summary
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
                f"(short {abs(delta)}) — check eligibility coverage"
            )
        elif t > 0 and delta > 14 and pref == 'baseline':
            warnings.append(
                f"{name}: served {total} vs target {t} "
                f"(+{delta} over) — baseline surgeon over target"
            )

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
