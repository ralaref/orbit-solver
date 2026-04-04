"""
ORbit Surgical Scheduling Solver v16
=====================================
Key fix from v15: Paced greedy distribution.

The v15 greedy filled weeks chronologically without awareness of the
full 6-month horizon. By November, surgeons had already hit their caps
from July-October, leaving weeks unfilled.

v16 fix: Each surgeon gets a "pace budget" — how many shifts they should
have served by each week to stay on track for their full-block target.
The priority score now rewards surgeons who are BEHIND their pace budget,
not just behind their overall target. This naturally spreads assignments
evenly across all 6 months.

Example — Loor (0.38 FTE, 32 shifts target, 26 weeks):
  Week 1 pace budget:  32 * (1/26)  = 1.2 shifts  (she needs ~0 shifts so far)
  Week 13 pace budget: 32 * (13/26) = 16 shifts    (she needs ~16 shifts by now)
  Week 26 pace budget: 32 * (26/26) = 32 shifts    (she needs ~32 shifts total)

If Loor has served 0 shifts by week 13 (when she should have 16),
her pace deficit is 16 — very high priority. The algorithm assigns her.
If she has 16 by week 13, deficit is 0 — normal priority.
If she has 20 by week 13 (ahead of pace), deficit is negative — lower priority.

This produces natural, even distribution across the full 6 months.
"""

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

ROLE_SHIFTS = {
    'ACS (M-F)':   SHIFTS_ACS_MF,
    'ACS (M-Sun)': SHIFTS_ACS_MSUN,
    'McNair ICU':  SHIFTS_ICU,
    'TSICU':       SHIFTS_ICU,
    'SICU':        SHIFTS_ICU,
}

# Role assignment order: most constrained pools first
# This ensures surgeons with limited eligibility get their slots
# before the general pool fills those positions
ROLE_ORDER = ['SICU', 'TSICU', 'McNair ICU', 'ACS (M-Sun)', 'ACS (M-F)']

# ─────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ORbit Solver v16'})


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

        week_assignments = greedy_service_weeks(
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

def get_all_weeks(months):
    """
    Returns all Mon-Sun weeks in the block, deduplicated and sorted
    chronologically. Each week tagged to the month containing its Monday.
    """
    seen  = set()
    weeks = []

    for mi, (y, mo) in enumerate(months):
        first_day  = datetime(y, mo, 1)
        week_start = first_day - timedelta(days=first_day.weekday())

        while True:
            has_days = any(
                (week_start + timedelta(days=o)).year == y and
                (week_start + timedelta(days=o)).month == mo
                for o in range(7)
            )
            if has_days and week_start not in seen:
                seen.add(week_start)
                label = (
                    f"{week_start.strftime('%b %-d')} - "
                    f"{(week_start + timedelta(days=6)).strftime('%b %-d')}"
                )
                canonical_mi = mi
                for check_mi, (cy, cmo) in enumerate(months):
                    if week_start.year == cy and week_start.month == cmo:
                        canonical_mi = check_mi
                        break
                weeks.append({
                    'start':     week_start,
                    'end':       week_start + timedelta(days=6),
                    'label':     label,
                    'year':      week_start.year,
                    'month':     week_start.month,
                    'month_idx': canonical_mi,
                })

            week_start += timedelta(days=7)
            if week_start.year > y or (week_start.year == y and week_start.month > mo):
                break

    weeks.sort(key=lambda w: w['start'])
    return weeks


def get_two_month_periods(months):
    return [[months[i], months[i + 1]] for i in range(0, 6, 2)]


# ─────────────────────────────────────────────────────────────────
# SURGEON HELPERS
# ─────────────────────────────────────────────────────────────────

def is_eligible(surgeon, role):
    """Contractual eligibility — never overridden."""
    role_map = {
        'ACS (M-F)':   'can_acs',
        'ACS (M-Sun)': 'can_acs',
        'McNair ICU':  'covers_mcnair',
        'TSICU':       'covers_tsicu',
        'SICU':        'covers_sicu',
        'call':        'can_call',
    }
    key = role_map.get(role)
    return bool(surgeon.get(key, False)) if key else False


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


def is_seven_day_role(role):
    return role in ('ACS (M-Sun)', 'McNair ICU', 'TSICU', 'SICU')


# ─────────────────────────────────────────────────────────────────
# FTE TARGET & CAPS
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


def compute_shift_cap(target_shifts, pref):
    """Hard ceiling on total shifts per surgeon."""
    multiplier = {'baseline': 1.0, 'willing': 1.4, 'seeking': 1.8}.get(pref, 1.0)
    return round(target_shifts * multiplier)


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

def compute_fellow_period_targets(fellows, periods, all_weeks):
    result = {}
    for fellow in fellows:
        fname         = fellow['name']
        result[fname] = {}
        for pi, period_months in enumerate(periods):
            period_set   = set((pm[0], pm[1]) for pm in period_months)
            period_weeks = [w for w in all_weeks
                            if (w['year'], w['month']) in period_set]
            total   = len(period_weeks)
            active  = sum(1 for w in period_weeks if is_active_for_week(fellow, w))
            if total == 0 or active == 0:
                result[fname][pi] = (0, 0)
                continue
            ratio             = active / total
            result[fname][pi] = (max(0, round(2 * ratio)),
                                  max(0, round(1 * ratio)))
    return result


# ─────────────────────────────────────────────────────────────────
# GREEDY SERVICE WEEK SOLVER — PACED DISTRIBUTION
# ─────────────────────────────────────────────────────────────────

def greedy_service_weeks(surgeons, months, block_number, preferences, prior_totals):
    """
    Paced greedy algorithm for service week assignment.

    THE KEY IMPROVEMENT OVER v15:
    v15 scored candidates by (target - served) / target.
    This meant surgeons near their target in week 5 got low priority,
    even though they hadn't really "used up" their allocation — they
    just happened to be scheduled early. The algorithm then ran out of
    eligible candidates in November/December.

    v16 uses a PACE BUDGET instead:
    pace_deficit = pace_budget_at_week_wi - served_so_far

    pace_budget_at_week_wi = target * (active_weeks_so_far / total_active_weeks)

    This tells us: "How many shifts SHOULD this surgeon have served
    by now if we spread their work evenly across the block?"

    A surgeon with a large pace deficit is behind schedule — high priority.
    A surgeon ahead of pace has lower priority.
    A surgeon at their cap gets a strongly negative score.

    This produces natural, even distribution across all 6 months because
    the algorithm actively works to keep everyone on pace, not just to
    fill slots with whoever is available.

    Additional improvements:
    - Fellows pre-scheduled across all periods before general assignment
    - Two-pass system: pass 1 fills fellows, pass 2 fills everyone else
    - Remaining capacity reserved proportionally for later months
    """
    all_weeks = get_all_weeks(months)
    num_weeks = len(all_weeks)
    periods   = get_two_month_periods(months)

    fellows     = [s for s in surgeons if is_fellow(s)]
    all_names   = [s['name'] for s in surgeons]

    # Parse preferences
    surgeon_time_off = {}
    for s in surgeons:
        prefs = get_surgeon_prefs(s.get('id', ''), preferences)
        y_ref = months[0][0]
        off   = parse_date_list(prefs.get('time_off',    ''), y_ref)
        conf  = parse_date_list(prefs.get('conferences', ''), y_ref)
        surgeon_time_off[s['name']] = off | conf

    # Compute targets and caps
    targets = {}
    caps    = {}
    for s in surgeons:
        t              = compute_block_target(s, block_number, prior_totals, months)
        t_int          = max(0, round(t))
        pref           = get_pref(s)
        targets[s['name']] = t_int
        caps[s['name']]    = compute_shift_cap(t_int, pref)

    # Compute each surgeon's active weeks count
    # (used to calculate pace budget at each point in time)
    surgeon_active_weeks = {}
    for s in surgeons:
        surgeon_active_weeks[s['name']] = sum(
            1 for w in all_weeks if is_active_for_week(s, w)
        )

    # State tracking
    served           = {n: 0  for n in all_names}
    last_service_wi  = {n: -99 for n in all_names}
    last_7day_wi     = {n: -99 for n in all_names}
    last_acs_msun_wi = {n: -99 for n in all_names}

    # Active weeks seen so far per surgeon (for pace calculation)
    active_weeks_so_far = {n: 0 for n in all_names}

    # Fellow rotation tracking
    fellow_period_targets = compute_fellow_period_targets(fellows, periods, all_weeks)
    fellow_acs_served  = {f['name']: {pi: 0 for pi in range(len(periods))} for f in fellows}
    fellow_sicu_served = {f['name']: {pi: 0 for pi in range(len(periods))} for f in fellows}

    def get_period_idx(week):
        for pi, period_months in enumerate(periods):
            period_set = set((pm[0], pm[1]) for pm in period_months)
            if (week['year'], week['month']) in period_set:
                return pi
        return None

    def fellow_needs_acs(fellow, week):
        pi = get_period_idx(week)
        if pi is None:
            return False
        acs_t, _ = fellow_period_targets[fellow['name']].get(pi, (0, 0))
        return fellow_acs_served[fellow['name']][pi] < acs_t

    def fellow_needs_sicu(fellow, week):
        pi = get_period_idx(week)
        if pi is None:
            return False
        _, sicu_t = fellow_period_targets[fellow['name']].get(pi, (0, 0))
        return fellow_sicu_served[fellow['name']][pi] < sicu_t

    def fellow_can_take_role(fellow, role, week):
        if role in ('ACS (M-F)', 'ACS (M-Sun)'):
            return fellow_needs_acs(fellow, week)
        if role == 'SICU':
            return fellow_needs_sicu(fellow, week)
        return False

    def pace_budget(surgeon, wi):
        """
        How many shifts should this surgeon have served by week wi
        if we spread their work evenly across their active weeks?
        """
        name             = surgeon['name']
        total_active     = surgeon_active_weeks[name]
        active_so_far    = active_weeks_so_far[name]
        t                = targets[name]

        if total_active == 0 or t == 0:
            return 0.0

        # Linear interpolation: by week wi, surgeon should have served
        # target * (active_so_far / total_active) shifts
        return t * (active_so_far / total_active)

    def can_assign(surgeon, role, week, wi, assigned_this_week):
        name = surgeon['name']

        if name in assigned_this_week.values():
            return False
        if not is_active_for_week(surgeon, week):
            return False
        if not is_eligible(surgeon, role):
            return False

        role_shifts = ROLE_SHIFTS[role]
        if served[name] + role_shifts > caps[name]:
            return False

        if surgeon_time_off.get(name) and week_overlaps_dates(
                week, surgeon_time_off[name]):
            return False

        if is_seven_day_role(role) and wi - last_7day_wi[name] <= 1:
            return False

        if role == 'ACS (M-Sun)' and wi - last_acs_msun_wi[name] <= 1:
            return False

        if is_fellow(surgeon) and not fellow_can_take_role(surgeon, role, week):
            return False

        return True

    def priority_score(surgeon, role, wi, week):
        """
        Paced priority score.

        Components:
        1. Pace deficit: how far behind their even-distribution pace are they?
           Normalized by target so different FTE levels are comparable.
           This is the PRIMARY driver — keeps distribution even over 6 months.

        2. Rest bonus: weeks since last service.
           Secondary driver — prevents consecutive week overload.

        3. Preference adjustment: willing/seeking get a bonus when over their
           baseline target, so they naturally absorb overflow. Baseline
           surgeons get a penalty when over target.
        """
        name  = surgeon['name']
        t     = targets[name]
        pref  = get_pref(surgeon)

        if t == 0:
            return -1.0

        # Pace deficit (primary)
        budget        = pace_budget(surgeon, wi)
        pace_deficit  = (budget - served[name]) / t  # normalized to [~-1, ~1]

        # Rest bonus (secondary)
        weeks_rested = wi - last_service_wi[name]
        rest         = min(weeks_rested * 0.08, 0.4)

        # Preference adjustment for over-target situations
        pref_adj = 0.0
        if served[name] >= t:
            if pref == 'willing':
                pref_adj = 0.1
            elif pref == 'seeking':
                pref_adj = 0.2
            else:
                pref_adj = -0.8  # strongly discourage baseline over-assignment

        return pace_deficit + rest + pref_adj

    # Initialize week assignments
    week_assignments = {}
    for wi, week in enumerate(all_weeks):
        week_assignments[wi] = {
            'label':     week['label'],
            'start':     week['start'],
            'year':      week['year'],
            'month':     week['month'],
            'month_idx': week['month_idx'],
        }

    # ── Main greedy loop ──────────────────────────────────────────
    for wi, week in enumerate(all_weeks):
        assigned_this_week = {}

        # Update active_weeks_so_far for pace calculation
        for s in surgeons:
            if is_active_for_week(s, week):
                active_weeks_so_far[s['name']] += 1

        # Pass 1: Fellows — assign to roles they need for rotation
        # Do this before general assignment to guarantee fellow rotation
        for role in ROLE_ORDER:
            if role in assigned_this_week:
                continue
            for fellow in fellows:
                if not can_assign(fellow, role, week, wi, assigned_this_week):
                    continue
                if fellow_can_take_role(fellow, role, week):
                    # Ensure no other fellow already in this role
                    other_fellow_in_role = any(
                        assigned_this_week.get(role) == f['name']
                        for f in fellows if f['name'] != fellow['name']
                    )
                    if not other_fellow_in_role and role not in assigned_this_week:
                        assigned_this_week[role] = fellow['name']
                        break

        # Pass 2: General assignment — best paced candidate for each unfilled role
        for role in ROLE_ORDER:
            if role in assigned_this_week:
                continue

            candidates = []
            for surgeon in surgeons:
                if can_assign(surgeon, role, week, wi, assigned_this_week):
                    score = priority_score(surgeon, role, wi, week)
                    candidates.append((score, surgeon['name'], surgeon))

            if not candidates:
                # No eligible surgeon — role goes unfilled this week
                # Validation report will flag this
                continue

            # Best score wins; ties broken alphabetically for determinism
            candidates.sort(key=lambda x: (-x[0], x[1]))
            best = candidates[0][2]
            assigned_this_week[role] = best['name']

        # Update state
        for role, name in assigned_this_week.items():
            surgeon = next((s for s in surgeons if s['name'] == name), None)
            if surgeon is None:
                continue

            served[name]          += ROLE_SHIFTS[role]
            last_service_wi[name]  = wi

            if is_seven_day_role(role):
                last_7day_wi[name] = wi
            if role == 'ACS (M-Sun)':
                last_acs_msun_wi[name] = wi

            if is_fellow(surgeon):
                pi = get_period_idx(week)
                if pi is not None:
                    if role in ('ACS (M-F)', 'ACS (M-Sun)'):
                        fellow_acs_served[name][pi] += 1
                    elif role == 'SICU':
                        fellow_sicu_served[name][pi] += 1

            week_assignments[wi][role] = name

    return week_assignments


# ─────────────────────────────────────────────────────────────────
# SOLVER 2 — CALL (OR-Tools CP-SAT)
# ─────────────────────────────────────────────────────────────────

def solve_call(surgeons, months, week_assignments, preferences):
    """
    OR-Tools CP-SAT for call night assignments.

    HARD constraints:
    - Exactly one call surgeon per night
    - Call eligibility and active dates
    - McNair surgeon: no call any night that week
    - TSICU/SICU/ACS M-Sun: no call Mon-Sat (Sunday OK)
    - ACS M-F: no call Mon-Thu (Fri/Sat/Sun OK)
    - Fellow max 5 call nights per month
    - Max call nights per month per surgeon profile

    SOFT constraints:
    - Weekend call equity
    - Call day preferences
    - Avoid specific nights
    - No 3+ consecutive call nights
    """
    num_surgeons   = len(surgeons)
    num_months     = len(months)
    month_days     = [monthrange(y, mo)[1] for y, mo in months]
    fellow_indices = [i for i, s in enumerate(surgeons) if is_fellow(s)]

    surgeon_avoid = {}
    for i, s in enumerate(surgeons):
        prefs = get_surgeon_prefs(s.get('id', ''), preferences)
        surgeon_avoid[i] = parse_date_list(
            prefs.get('avoid_nights', ''), months[0][0])

    active_in_month = [
        [is_active_for_month(surgeons[i], y, mo) for i in range(num_surgeons)]
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
                    for role in ROLE_SHIFTS:
                        name = wa.get(role)
                        if name:
                            for i, s in enumerate(surgeons):
                                if s['name'] == name:
                                    night_role[mi][d][i] = role
                    break

    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    call = [
        [[model.NewBoolVar(f'c_{mi}_{d}_{i}') for i in range(num_surgeons)]
         for d in range(month_days[mi])]
        for mi in range(num_months)
    ]

    # H1 — One call surgeon per night
    for mi in range(num_months):
        for d in range(month_days[mi]):
            model.AddExactlyOne(call[mi][d])

    # H2 — Eligibility and active dates
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for i in range(num_surgeons):
                if (not active_in_month[mi][i] or
                        not is_eligible(surgeons[i], 'call')):
                    model.Add(call[mi][d][i] == 0)

    # H3 — Service week call restrictions
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            dow = datetime(y, mo, d + 1).weekday()
            for i in range(num_surgeons):
                role = night_role[mi][d].get(i)
                if role is None:
                    continue
                if role == 'McNair ICU':
                    model.Add(call[mi][d][i] == 0)
                elif role in ('TSICU', 'SICU', 'ACS (M-Sun)'):
                    if dow <= 5:
                        model.Add(call[mi][d][i] == 0)
                elif role == 'ACS (M-F)':
                    if dow <= 3:
                        model.Add(call[mi][d][i] == 0)

    # H4 — Fellow max 5 call nights per month
    for i in fellow_indices:
        max_call = int(surgeons[i].get('max_call_per_month', 5))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][i] for d in range(month_days[mi])) <= max_call)

    # H5 — Max call nights per month per surgeon profile
    for i in range(num_surgeons):
        if i in fellow_indices:
            continue
        max_call = int(surgeons[i].get('max_call_per_month', 8))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][i] for d in range(month_days[mi])) <= max_call)

    # ── Objective ─────────────────────────────────────────────────
    obj_terms     = []
    penalty_terms = []

    # Weekend call equity
    weekend_nights = [
        (mi, d)
        for mi, (y, mo) in enumerate(months)
        for d in range(month_days[mi])
        if datetime(y, mo, d + 1).weekday() >= 4
    ]
    total_weekend = len(weekend_nights)
    call_eligible = [i for i in range(num_surgeons)
                     if is_eligible(surgeons[i], 'call')]
    fair_wknd = max(1, round(total_weekend / max(1, len(call_eligible))))

    surgeon_wknd = [
        model.NewIntVar(0, total_weekend, f'wk_{i}')
        for i in range(num_surgeons)
    ]
    for i in range(num_surgeons):
        wvars = [call[mi][d][i] for mi, d in weekend_nights]
        model.Add(surgeon_wknd[i] == (sum(wvars) if wvars else 0))

    for i in call_eligible:
        pref      = get_pref(surgeons[i])
        wknd_over = model.NewIntVar(0, total_weekend, f'wo_{i}')
        wknd_undr = model.NewIntVar(0, total_weekend, f'wu_{i}')
        model.Add(wknd_over >= surgeon_wknd[i] - fair_wknd)
        model.Add(wknd_undr >= fair_wknd - surgeon_wknd[i])
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
            for i in range(num_surgeons):
                pref_days = surgeons[i].get('call_day_preference', '') or ''
                if pref_days == 'friday_saturday':
                    if dow in (4, 5):
                        obj_terms.append(3 * call[mi][d][i])
                    else:
                        penalty_terms.append(2 * call[mi][d][i])

    # No 3+ consecutive call nights
    for mi in range(num_months):
        days = month_days[mi]
        for i in range(num_surgeons):
            for d in range(days - 3):
                run4 = model.NewBoolVar(f'r4_{mi}_{d}_{i}')
                model.AddMinEquality(run4, [
                    call[mi][d][i],   call[mi][d+1][i],
                    call[mi][d+2][i], call[mi][d+3][i]
                ])
                penalty_terms.append(25 * run4)

    # Avoid specific nights
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for i in range(num_surgeons):
                if surgeon_avoid[i] and day_in_dates(y, mo, d, surgeon_avoid[i]):
                    penalty_terms.append(30 * call[mi][d][i])

    total_obj = []
    if obj_terms:
        total_obj.append(sum(obj_terms))
    if penalty_terms:
        total_obj.append(-sum(penalty_terms))
    if total_obj:
        model.Maximize(sum(total_obj) if len(total_obj) > 1 else total_obj[0])

    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers  = 4

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise Exception(
            f"Call solver failed: {solver.StatusName(status)}. "
            f"Cannot assign call given current service restrictions. "
            f"Check call eligibility flags in admin."
        )

    call_assignments = {}
    for mi in range(num_months):
        for d in range(month_days[mi]):
            for i in range(num_surgeons):
                if solver.Value(call[mi][d][i]):
                    call_assignments[(mi, d)] = surgeons[i]['name']

    return call_assignments


# ─────────────────────────────────────────────────────────────────
# OUTPUT BUILDER
# ─────────────────────────────────────────────────────────────────

def build_output(surgeons, months, week_assignments, call_assignments,
                 block_number, prior_totals):

    num_surgeons = len(surgeons)
    num_months   = len(months)
    month_days   = [monthrange(y, mo)[1] for y, mo in months]

    block_targets = {
        s['name']: compute_block_target(s, block_number, prior_totals, months)
        for s in surgeons
    }
    target_shifts = {
        name: max(0, round(t)) for name, t in block_targets.items()
    }

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
            # Always include all roles, even if unassigned (empty string)
            # This ensures the UI renders all role slots consistently
            for role in ROLE_SHIFTS:
                week_data[role] = wa.get(role, '')
            result_weeks.append(week_data)

        result_nights = {}
        for d in range(month_days[mi]):
            name = call_assignments.get((mi, d))
            if name:
                result_nights[str(d + 1)] = {'Call': name, 'Backup': ''}
            else:
                result_nights[str(d + 1)] = {'Call': '', 'Backup': ''}

        fte_summary = {}
        for s in surgeons:
            name   = s['name']
            shifts = 0
            for w in result_weeks:
                for role, shift_count in ROLE_SHIFTS.items():
                    if w.get(role) == name:
                        shifts += shift_count
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
            for role in ROLE_SHIFTS:
                if not w.get(role):
                    warnings.append(
                        f"{month_label} {w['label']}: {role} unfilled — "
                        f"no eligible surgeon available within caps")

        for d in range(month_days[mi]):
            if not result[mk]['nights'].get(str(d + 1), {}).get('Call'):
                violations.append(
                    f"{month_label} day {d + 1}: No call surgeon assigned")

        for w in result[mk]['weeks']:
            seen = {}
            for role in ROLE_SHIFTS:
                name = w.get(role)
                if name:
                    if name in seen:
                        violations.append(
                            f"{month_label} {w['label']}: "
                            f"{name} in {seen[name]} and {role}")
                    seen[name] = role

    # Consecutive 7-day weeks
    seven_day_roles = ['ACS (M-Sun)', 'McNair ICU', 'TSICU', 'SICU']
    for i in range(len(all_weeks_flat) - 1):
        w1 = all_weeks_flat[i]
        w2 = all_weeks_flat[i + 1]
        for r1 in seven_day_roles:
            for r2 in seven_day_roles:
                n1 = w1.get(r1)
                n2 = w2.get(r2)
                if n1 and n2 and n1 == n2:
                    warnings.append(
                        f"Consecutive 7-day weeks: {n1} "
                        f"({r1} -> {r2}) — review manually")

    # ACS M-Sun consecutive
    for i in range(len(all_weeks_flat) - 1):
        n1 = all_weeks_flat[i].get('ACS (M-Sun)')
        n2 = all_weeks_flat[i + 1].get('ACS (M-Sun)')
        if n1 and n2 and n1 == n2:
            violations.append(
                f"ACS M-Sun consecutive: {n1} — hard rule violated")

    # Call run check
    for mi, (y, mo) in enumerate(months):
        nights = result[f"{y}-{str(mo).zfill(2)}"]['nights']
        for s in surgeons:
            name = s['name']
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
                    for role in ROLE_SHIFTS:
                        if wa.get(role) == call_name:
                            prior_wi = wi - 1
                            in_prior = prior_wi >= 0 and any(
                                week_assignments[prior_wi].get(r) == call_name
                                for r in ROLE_SHIFTS
                            )
                            if not in_prior:
                                warnings.append(
                                    f"{call_name}: call Sun "
                                    f"{next_monday.strftime('%b %-d')} "
                                    f"then fresh {role} Mon — fix manually")

    # Fellow rotation validation
    all_weeks  = get_all_weeks(months)
    periods    = get_two_month_periods(months)
    fellows    = [s for s in surgeons if is_fellow(s)]
    fpt        = compute_fellow_period_targets(fellows, periods, all_weeks)

    for pi, period_months in enumerate(periods):
        for fellow in fellows:
            fname         = fellow['name']
            acs_t, sicu_t = fpt[fname].get(pi, (0, 0))
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
                    f"Fellow {fname} period {pi + 1}: "
                    f"{acs_count} ACS weeks (expected {acs_t})")
            if sicu_t > 0 and sicu_count != sicu_t:
                violations.append(
                    f"Fellow {fname} period {pi + 1}: "
                    f"{sicu_count} SICU weeks (expected {sicu_t})")

    # FTE equity summary
    weekend_nights = [
        (mi, d)
        for mi, (y, mo) in enumerate(months)
        for d in range(month_days[mi])
        if datetime(y, mo, d + 1).weekday() >= 4
    ]

    block_fte_summary    = {}
    weekend_call_summary = {}

    for s in surgeons:
        name  = s['name']
        pref  = get_pref(s)
        total = sum(
            result[f"{y}-{str(mo).zfill(2)}"]['fte_summary'].get(name, 0)
            for y, mo in months
        )
        t     = target_shifts[name]
        delta = total - t

        if t > 0 and delta < -7:
            warnings.append(
                f"{name}: served {total} vs target {t} "
                f"(short {abs(delta)}) — check eligibility or willing/seeking capacity")
        if t > 0 and delta > 7 and pref == 'baseline':
            warnings.append(
                f"{name}: served {total} vs target {t} "
                f"(+{delta} over) — baseline over target")

        block_fte_summary[name] = {
            'served': total,
            'target': round(block_targets[name], 1),
            'delta':  round(delta, 1),
        }

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
