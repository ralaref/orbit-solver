"""
ORbit Surgical Scheduling Solver v15
=====================================
Architecture:
  - Greedy algorithm for service week assignments
    (deterministic, fast, human-like, always produces fair results)
  - OR-Tools CP-SAT for call night assignments
    (constraint satisfaction for the complex nightly interaction problem)

Greedy philosophy:
  For each week, for each role, pick the best available surgeon using
  a priority score that rewards surgeons who:
    1. Are furthest below their FTE target (proportionally)
    2. Have not worked recently (avoids consecutive weeks)
    3. Match the role's eligibility requirements

  This exactly mirrors how a human scheduler thinks:
  "Who needs this week most? Who hasn't worked in a while?"

Rules honored:
  - FTE targets: 84 x FTE per 6-month block (prorated for start/end dates)
  - Eligibility flags: contractual, never overridden
  - Active dates: all surgeons treated identically
  - No consecutive 7-day service weeks (hard rule in greedy)
  - ACS M-Sun no repeat consecutive weeks (hard rule in greedy)
  - Fellow rotation: 2 ACS + 1 SICU per 2-month period
  - Fellows cannot share same role same week
  - Baseline surgeons stop at their FTE target
  - Willing surgeons can absorb up to 40% overflow
  - Seeking surgeons can absorb up to 80% overflow
  - Surgeon preferences (time off, conferences) respected
  - Call restrictions enforced by Solver 2
  - Weekend call equity enforced by Solver 2
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from calendar import monthrange
import os

app = Flask(__name__)
CORS(app)

# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# CONSTANTS
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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

# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# HEALTH
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ORbit Solver v15'})


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# ENDPOINT
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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

        # Step 1: Greedy service week assignment
        week_assignments = greedy_service_weeks(
            surgeons=surgeons,
            months=months,
            block_number=block_number,
            preferences=preferences,
            prior_totals=prior_totals,
        )

        # Step 2: OR-Tools call assignment
        call_assignments = solve_call(
            surgeons=surgeons,
            months=months,
            week_assignments=week_assignments,
            preferences=preferences,
        )

        # Step 3: Build output and validation report
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


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# WEEK CALCULATION
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def get_all_weeks(months):
    """
    Returns a deduplicated, chronologically sorted list of all
    Mon-Sun weeks in the block. Each week appears exactly once,
    tagged to the month containing its Monday.
    """
    seen   = set()
    weeks  = []

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


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# SURGEON HELPERS
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def is_eligible(surgeon, role):
    """Contractual eligibility â never overridden."""
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
    """
    All surgeons treated identically â active on and after start_date,
    inactive after departure_date. No special cases.
    """
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


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# FTE TARGET & CAPS
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def compute_block_target(surgeon, block_number, prior_totals, months):
    """
    Block 1: target = 84 x FTE
    Block 2: target = (168 x FTE) - Block 1 actuals
    Prorated for start/departure dates within the block.
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
    """
    Maximum total service shifts a surgeon can receive.
    baseline: exactly their target
    willing:  up to 140% of target
    seeking:  up to 180% of target
    """
    multiplier = {'baseline': 1.0, 'willing': 1.4, 'seeking': 1.8}.get(pref, 1.0)
    return round(target_shifts * multiplier)


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# PREFERENCE PARSING
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# FELLOW ROTATION TRACKING
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def compute_fellow_period_targets(fellows, periods, all_weeks):
    """
    For each fellow and each 2-month period, compute how many
    ACS weeks and SICU weeks they need to fulfill their rotation.
    Returns: {fellow_name: {period_idx: (acs_target, sicu_target)}}
    """
    result = {}
    for fellow in fellows:
        fname  = fellow['name']
        result[fname] = {}
        for pi, period_months in enumerate(periods):
            period_set   = set((pm[0], pm[1]) for pm in period_months)
            period_weeks = [w for w in all_weeks
                            if (w['year'], w['month']) in period_set]
            total   = len(period_weeks)
            active  = sum(1 for w in period_weeks
                          if is_active_for_week(fellow, w))
            if total == 0 or active == 0:
                result[fname][pi] = (0, 0)
                continue
            ratio = active / total
            result[fname][pi] = (max(0, round(2 * ratio)),
                                  max(0, round(1 * ratio)))
    return result


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# GREEDY SERVICE WEEK SOLVER
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def greedy_service_weeks(surgeons, months, block_number, preferences, prior_totals):
    """
    Greedy algorithm for service week assignment.

    For each week in chronological order, for each role, select the
    best available surgeon using a priority score. This mirrors exactly
    how a human scheduler thinks.

    PRIORITY SCORE (higher = more likely to be assigned):
      base     = (target - served) / target   [0..1] how far below target
      recency  = weeks_since_last_service * 0.1  [rest bonus]
      pref_adj = 0.2 bonus for willing/seeking when above baseline target

    Constraints checked before scoring:
      - Surgeon is active this week
      - Surgeon is eligible for this role
      - Surgeon has not hit their shift cap
      - Surgeon is not already assigned to another role this week
      - No consecutive 7-day service week (ACS M-Sun, McNair, TSICU, SICU)
      - ACS M-Sun not assigned two consecutive weeks
      - Surgeon not on time off / conference this week
      - Fellow rotation constraints

    Fellow rotation is pre-planned before the greedy loop:
      We calculate exactly which weeks each fellow needs ACS and SICU
      then reserve those slots before filling the rest.
    """

    all_weeks = get_all_weeks(months)
    periods   = get_two_month_periods(months)

    fellows       = [s for s in surgeons if is_fellow(s)]
    non_fellows   = [s for s in surgeons if not is_fellow(s)]
    all_surgeons  = surgeons  # full list for indexing

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
        t           = compute_block_target(s, block_number, prior_totals, months)
        t_int       = max(0, round(t))
        pref        = get_pref(s)
        targets[s['name']] = t_int
        caps[s['name']]    = compute_shift_cap(t_int, pref)

    # State tracking
    served          = {s['name']: 0 for s in surgeons}       # shifts served so far
    last_service_wi = {s['name']: -99 for s in surgeons}     # last week index with any service
    last_acs_msun_wi = {s['name']: -99 for s in surgeons}    # last ACS M-Sun week index
    last_7day_wi    = {s['name']: -99 for s in surgeons}     # last 7-day role week index

    # week_assignments[wi] = {role: surgeon_name}
    week_assignments = {}
    for wi, week in enumerate(all_weeks):
        week_assignments[wi] = {
            'label':     week['label'],
            'start':     week['start'],
            'year':      week['year'],
            'month':     week['month'],
            'month_idx': week['month_idx'],
        }

    # ââ Pre-plan fellow rotation ââââââââââââââââââââââââââââââââââ
    # Calculate fellow rotation targets per period
    fellow_period_targets = compute_fellow_period_targets(fellows, periods, all_weeks)

    # Track fellow rotation fulfillment
    fellow_acs_served  = {f['name']: {pi: 0 for pi in range(len(periods))} for f in fellows}
    fellow_sicu_served = {f['name']: {pi: 0 for pi in range(len(periods))} for f in fellows}

    def get_period_idx(week):
        period_set_list = [
            set((pm[0], pm[1]) for pm in p) for p in periods
        ]
        for pi, pset in enumerate(period_set_list):
            if (week['year'], week['month']) in pset:
                return pi
        return None

    def fellow_needs_acs(fellow, week):
        """Does this fellow still need an ACS week in this period?"""
        pi = get_period_idx(week)
        if pi is None:
            return False
        acs_t, _ = fellow_period_targets[fellow['name']].get(pi, (0, 0))
        return fellow_acs_served[fellow['name']][pi] < acs_t

    def fellow_needs_sicu(fellow, week):
        """Does this fellow still need a SICU week in this period?"""
        pi = get_period_idx(week)
        if pi is None:
            return False
        _, sicu_t = fellow_period_targets[fellow['name']].get(pi, (0, 0))
        return fellow_sicu_served[fellow['name']][pi] < sicu_t

    def fellow_can_take_role(fellow, role, week):
        """Check if fellow needs this role type in this period."""
        if role in ('ACS (M-F)', 'ACS (M-Sun)'):
            return fellow_needs_acs(fellow, week)
        if role == 'SICU':
            return fellow_needs_sicu(fellow, week)
        return False  # Fellows don't take McNair or TSICU

    # ââ Candidate check âââââââââââââââââââââââââââââââââââââââââââ

    def is_seven_day_role(role):
        return role in ('ACS (M-Sun)', 'McNair ICU', 'TSICU', 'SICU')

    def can_assign(surgeon, role, week, wi, assigned_this_week):
        """
        Returns True if this surgeon can take this role this week.
        Checks all hard constraints.
        """
        name = surgeon['name']

        # Already assigned to another role this week
        if name in assigned_this_week.values():
            return False

        # Not active this week
        if not is_active_for_week(surgeon, week):
            return False

        # Not eligible for this role (contractual)
        if not is_eligible(surgeon, role):
            return False

        # Hit shift cap
        role_shifts = ROLE_SHIFTS[role]
        if served[name] + role_shifts > caps[name]:
            return False

        # On time off or conference this week
        if surgeon_time_off.get(name) and week_overlaps_dates(
                week, surgeon_time_off[name]):
            return False

        # No consecutive 7-day service weeks
        if is_seven_day_role(role) and (wi - last_7day_wi[name]) <= 1:
            return False

        # ACS M-Sun cannot repeat consecutive weeks
        if role == 'ACS (M-Sun)' and (wi - last_acs_msun_wi[name]) <= 1:
            return False

        # Fellow-specific: only assign fellows to roles they need for rotation
        if is_fellow(surgeon):
            if not fellow_can_take_role(surgeon, role, week):
                return False
            # Only one fellow per role per week
            for existing_name in assigned_this_week.values():
                for f in fellows:
                    if f['name'] == existing_name and existing_name != name:
                        existing_role = [
                            r for r, n in assigned_this_week.items() if n == existing_name
                        ]
                        if existing_role and existing_role[0] == role:
                            return False

        return True

    def priority_score(surgeon, role, wi):
        """
        Priority score â higher means this surgeon should get this week.

        The score has three components:
        1. Need: how far below their target are they? (0-1 scale)
           A surgeon at 0% of target scores 1.0 (maximum need)
           A surgeon at 100% of target scores 0.0 (no need)
        2. Rest: how long since they last worked? (rewards rest)
           Each week since last service adds 0.1 to score
        3. Preference: willing/seeking get a small bonus after target
           so they naturally absorb overflow

        This score means the surgeon who NEEDS work the most and
        has RESTED the longest always gets priority. Exactly like
        a human scheduler would think.
        """
        name = surgeon['name']
        t    = targets[name]

        if t == 0:
            # Zero-target surgeon only gets assigned if no one else can
            need = 0.0
        else:
            # Proportional need: (target - served) / target
            # Negative if over target (disincentivizes over-assignment)
            need = (t - served[name]) / t

        # Rest bonus: weeks since last service
        weeks_rested = wi - last_service_wi[name]
        rest = min(weeks_rested * 0.1, 0.5)  # cap at 0.5 to keep need dominant

        # Preference adjustment
        pref     = get_pref(surgeon)
        pref_adj = 0.0
        if served[name] >= t:
            # Over target â willing/seeking get small bonus, baseline gets penalty
            if pref == 'willing':
                pref_adj = 0.1
            elif pref == 'seeking':
                pref_adj = 0.2
            else:
                pref_adj = -0.5  # strongly discourage baseline over-assignment

        return need + rest + pref_adj

    # ââ ROLES in assignment order âââââââââââââââââââââââââââââââââ
    # Order matters for fairness:
    # 1. SICU first (fewest eligible surgeons â most constrained)
    # 2. TSICU second (also constrained â Chatterjee, Bonville, Lim, others)
    # 3. McNair third (moderate eligibility)
    # 4. ACS M-Sun fourth (moderate eligibility, no-repeat constraint)
    # 5. ACS M-F last (most flexible, Perez can only do this)
    role_order = ['SICU', 'TSICU', 'McNair ICU', 'ACS (M-Sun)', 'ACS (M-F)']

    # ââ Main greedy loop ââââââââââââââââââââââââââââââââââââââââââ
    for wi, week in enumerate(all_weeks):
        assigned_this_week = {}  # role -> surgeon_name for this week

        # First pass: try to assign fellows to roles they need for rotation
        # This ensures fellow rotation requirements are met before
        # the general pool fills up those slots
        for role in role_order:
            for fellow in fellows:
                if not can_assign(fellow, role, week, wi, assigned_this_week):
                    continue
                if fellow_can_take_role(fellow, role, week):
                    # Check no other fellow already assigned to this role
                    already = False
                    for f2 in fellows:
                        if assigned_this_week.get(role) == f2['name']:
                            already = True
                            break
                    if not already and role not in assigned_this_week:
                        assigned_this_week[role] = fellow['name']
                        # Only assign one fellow per role â break after first match
                        break

        # Second pass: fill remaining roles with best available surgeon
        for role in role_order:
            if role in assigned_this_week:
                continue  # Already filled by fellow

            # Find all candidates and score them
            candidates = []
            for surgeon in surgeons:
                if can_assign(surgeon, role, week, wi, assigned_this_week):
                    score = priority_score(surgeon, role, wi)
                    candidates.append((score, surgeon['name'], surgeon))

            if not candidates:
                # No eligible surgeon found â this should be rare
                # Log it but don't crash â validation will flag it
                continue

            # Sort by score descending, break ties by name for determinism
            candidates.sort(key=lambda x: (-x[0], x[1]))
            best_surgeon = candidates[0][2]
            assigned_this_week[role] = best_surgeon['name']

        # Update state from this week's assignments
        for role, name in assigned_this_week.items():
            # Find surgeon object
            surgeon = next((s for s in surgeons if s['name'] == name), None)
            if surgeon is None:
                continue

            role_shifts = ROLE_SHIFTS[role]
            served[name] += role_shifts
            last_service_wi[name] = wi

            if is_seven_day_role(role):
                last_7day_wi[name] = wi
            if role == 'ACS (M-Sun)':
                last_acs_msun_wi[name] = wi

            # Update fellow rotation tracking
            if is_fellow(surgeon):
                pi = get_period_idx(week)
                if pi is not None:
                    if role in ('ACS (M-F)', 'ACS (M-Sun)'):
                        fellow_acs_served[name][pi] += 1
                    elif role == 'SICU':
                        fellow_sicu_served[name][pi] += 1

            # Store in week_assignments
            week_assignments[wi][role] = name

    return week_assignments


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# SOLVER 2 â CALL (OR-Tools)
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def solve_call(surgeons, months, week_assignments, preferences):
    """
    OR-Tools CP-SAT solver for call night assignments.

    Takes the completed greedy service week schedule as fixed input.
    Assigns exactly one call surgeon per night.

    OR-Tools is appropriate here because call assignment has complex
    nightly interactions: restrictions depend on which surgeon is on
    service that specific night, weekend equity requires tracking
    totals across all nights, and avoid-night preferences interact
    with service restrictions in non-trivial ways.

    HARD constraints:
    - Exactly one call surgeon per night
    - Call eligibility (can_call flag)
    - Active dates
    - McNair surgeon: no call any night that week
    - TSICU/SICU/ACS M-Sun: no call Mon-Sat (Sunday OK)
    - ACS M-F: no call Mon-Thu (Fri/Sat/Sun OK)
    - Fellow max 5 call nights per month

    SOFT constraints (objective):
    - Weekend call equity (fair share per eligible surgeon)
    - Call day preferences (e.g. Rojas-Khalil Fri/Sat preferred)
    - Avoid specific nights from surgeon preferences
    - No 3+ consecutive call nights
    """
    num_surgeons   = len(surgeons)
    num_months     = len(months)
    month_days     = [monthrange(y, mo)[1] for y, mo in months]
    fellow_indices = [i for i, s in enumerate(surgeons) if is_fellow(s)]

    # Parse avoid-night preferences
    surgeon_avoid = {}
    for i, s in enumerate(surgeons):
        prefs = get_surgeon_prefs(s.get('id', ''), preferences)
        surgeon_avoid[i] = parse_date_list(
            prefs.get('avoid_nights', ''), months[0][0])

    # Active status per surgeon per month
    active_in_month = [
        [is_active_for_month(surgeons[i], y, mo) for i in range(num_surgeons)]
        for mi, (y, mo) in enumerate(months)
    ]

    # Build night -> service role lookup from week_assignments
    # For each night, which surgeons are on which service role?
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

    # call[mi][d][i] = 1 if surgeon i is on call on night (mi, d)
    call = [
        [[model.NewBoolVar(f'c_{mi}_{d}_{i}') for i in range(num_surgeons)]
         for d in range(month_days[mi])]
        for mi in range(num_months)
    ]

    # H1 â Exactly one call surgeon per night
    for mi in range(num_months):
        for d in range(month_days[mi]):
            model.AddExactlyOne(call[mi][d])

    # H2 â Call eligibility and active dates
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for i in range(num_surgeons):
                if (not active_in_month[mi][i] or
                        not is_eligible(surgeons[i], 'call')):
                    model.Add(call[mi][d][i] == 0)

    # H3 â Service week call restrictions
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            dow = datetime(y, mo, d + 1).weekday()  # 0=Mon, 6=Sun
            for i in range(num_surgeons):
                role = night_role[mi][d].get(i)
                if role is None:
                    continue
                if role == 'McNair ICU':
                    model.Add(call[mi][d][i] == 0)
                elif role in ('TSICU', 'SICU', 'ACS (M-Sun)'):
                    if dow <= 5:  # Mon-Sat blocked, Sunday OK
                        model.Add(call[mi][d][i] == 0)
                elif role == 'ACS (M-F)':
                    if dow <= 3:  # Mon-Thu blocked, Fri/Sat/Sun OK
                        model.Add(call[mi][d][i] == 0)

    # H4 â Fellow max 5 call nights per month
    for i in fellow_indices:
        max_call = int(surgeons[i].get('max_call_per_month', 5))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][i] for d in range(month_days[mi])) <= max_call)

    # H5 â Max call nights per month per surgeon (from profile)
    for i in range(num_surgeons):
        if i in fellow_indices:
            continue
        max_call = int(surgeons[i].get('max_call_per_month', 8))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][i] for d in range(month_days[mi])) <= max_call)

    # ââ Objective âââââââââââââââââââââââââââââââââââââââââââââââââ
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

    # Call day preference (e.g. Rojas-Khalil prefers Fri/Sat)
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
            f"Cannot assign call given current service week restrictions. "
            f"Check call eligibility flags in admin."
        )

    call_assignments = {}
    for mi in range(num_months):
        for d in range(month_days[mi]):
            for i in range(num_surgeons):
                if solver.Value(call[mi][d][i]):
                    call_assignments[(mi, d)] = surgeons[i]['name']

    return call_assignments


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# OUTPUT BUILDER
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def build_output(surgeons, months, week_assignments, call_assignments,
                 block_number, prior_totals):
    """
    Combines greedy service weeks and OR-Tools call into the standard
    JSON format expected by the Next.js frontend. Runs a comprehensive
    validation report.
    """
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
            for role in ROLE_SHIFTS:
                if role in wa:
                    week_data[role] = wa[role]
            result_weeks.append(week_data)

        result_nights = {}
        for d in range(month_days[mi]):
            name = call_assignments.get((mi, d))
            if name:
                result_nights[str(d + 1)] = {'Call': name, 'Backup': ''}

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

    # ââ Validation Report âââââââââââââââââââââââââââââââââââââââââ
    violations = []
    warnings   = []

    all_weeks_flat = []
    for mi in range(num_months):
        mk = f"{months[mi][0]}-{str(months[mi][1]).zfill(2)}"
        all_weeks_flat.extend(result[mk]['weeks'])

    for mi, (y, mo) in enumerate(months):
        mk          = f"{y}-{str(mo).zfill(2)}"
        month_label = datetime(y, mo, 1).strftime('%B %Y')

        # All roles assigned every week
        for w in result[mk]['weeks']:
            for role in ROLE_SHIFTS:
                if role not in w:
                    warnings.append(
                        f"{month_label} {w['label']}: {role} not assigned "
                        f"â no eligible surgeon available")

        # All nights covered
        for d in range(month_days[mi]):
            if str(d + 1) not in result[mk]['nights']:
                violations.append(
                    f"{month_label} day {d + 1}: No call surgeon assigned")

        # No surgeon in two roles simultaneously
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

    # Consecutive 7-day service weeks
    seven_day_roles = ['ACS (M-Sun)', 'McNair ICU', 'TSICU', 'SICU']
    for i in range(len(all_weeks_flat) - 1):
        w1 = all_weeks_flat[i]
        w2 = all_weeks_flat[i + 1]
        for r1 in seven_day_roles:
            for r2 in seven_day_roles:
                n1 = w1.get(r1)
                n2 = w2.get(r2)
                if n1 and n1 == n2:
                    warnings.append(
                        f"Consecutive 7-day weeks: {n1} "
                        f"({r1} -> {r2}) â review manually")

    # ACS M-Sun consecutive (should never happen with greedy)
    for i in range(len(all_weeks_flat) - 1):
        n1 = all_weeks_flat[i].get('ACS (M-Sun)')
        n2 = all_weeks_flat[i + 1].get('ACS (M-Sun)')
        if n1 and n1 == n2:
            violations.append(
                f"ACS M-Sun consecutive: {n1} â hard rule violated")

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

    # Sunday call -> Monday fresh service start
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
                            prior_wi  = wi - 1
                            in_prior  = prior_wi >= 0 and any(
                                week_assignments[prior_wi].get(r) == call_name
                                for r in ROLE_SHIFTS
                            )
                            if not in_prior:
                                warnings.append(
                                    f"{call_name}: call Sun "
                                    f"{next_monday.strftime('%b %-d')} "
                                    f"then fresh {role} Mon â fix manually")

    # Fellow rotation validation
    all_weeks = get_all_weeks(months)
    periods   = get_two_month_periods(months)
    fellows   = [s for s in surgeons if is_fellow(s)]
    fellow_period_targets = compute_fellow_period_targets(fellows, periods, all_weeks)

    for pi, period_months in enumerate(periods):
        for fellow in fellows:
            fname        = fellow['name']
            acs_t, sicu_t = fellow_period_targets[fname].get(pi, (0, 0))
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

        # Flag significant under-assignment
        if t > 0 and delta < -7:
            warnings.append(
                f"{name}: served {total} vs target {t} "
                f"(short {abs(delta)}) â insufficient eligible weeks available")

        # Flag baseline over-assignment
        if t > 0 and delta > 7 and pref == 'baseline':
            warnings.append(
                f"{name}: served {total} vs target {t} "
                f"(+{delta} over) â baseline surgeon over target")

        block_fte_summary[name] = {
            'served': total,
            'target': round(block_targets[name], 1),
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


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# RUN
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
