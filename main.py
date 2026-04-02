from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from calendar import monthrange
import os

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS — Universal scheduling rules, never changes
# ─────────────────────────────────────────────────────────────────

ANNUAL_FTE_SHIFTS = 168        # 1.0 FTE = 168 service shifts/year
BLOCK_FTE_SHIFTS = 84          # 1.0 FTE = 84 service shifts per 6-month block
SHIFTS_ACS_MF = 5              # ACS M-F week = 5 FTE shifts
SHIFTS_ACS_MSUN = 7            # ACS M-Sun week = 7 FTE shifts
SHIFTS_ICU = 7                 # Any ICU week = 7 FTE shifts
SHIFTS_CALL = 0                # Call nights = 0 FTE shifts
SHIFTS_BACKUP = 0              # Backup = 0 FTE shifts
MAX_SERVICE_SHIFTS_PER_MONTH = 14  # Hard cap for baseline surgeons

# Major holidays per block
HOLIDAYS_BLOCK1 = [
    'july_4th', 'labor_day', 'thanksgiving', 'christmas', 'new_years'
]
HOLIDAYS_BLOCK2 = [
    'mlk', 'presidents_day', 'easter', 'memorial_day'
]

# Block definitions
BLOCK1_MONTHS = [7, 8, 9, 10, 11, 12]   # Jul-Dec
BLOCK2_MONTHS = [1, 2, 3, 4, 5, 6]      # Jan-Jun

# ─────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ORbit Solver v2'})


# ─────────────────────────────────────────────────────────────────
# MAIN ENDPOINT — Full 6-month block solve
# ─────────────────────────────────────────────────────────────────

@app.route('/solve-block', methods=['POST'])
def solve_block():
    try:
        data = request.json

        # Required inputs
        surgeons     = data.get('surgeons', [])
        block_number = data.get('block_number', 1)   # 1=Jul-Dec, 2=Jan-Jun
        start_year   = data.get('start_year')         # e.g. 2026 for Jul-Dec 2026
        preferences  = data.get('preferences', [])
        prior_totals = data.get('prior_totals', {})   # Block 1 actuals, for Block 2 only

        if not surgeons:
            return jsonify({'success': False, 'error': 'No surgeons provided'}), 400
        if not start_year:
            return jsonify({'success': False, 'error': 'start_year required'}), 400

        # Determine which months to solve
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
            start_year=start_year,
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
    """
    Returns all Mon-Sun weeks that overlap with the given month.
    Label shows actual week dates (may span two months).
    """
    first_day = datetime(year, month, 1)
    dow = first_day.weekday()  # 0=Mon
    week_start = first_day - timedelta(days=dow)

    weeks = []
    while True:
        week_end = week_start + timedelta(days=6)

        # Which days of THIS month fall in this week
        days_in_week = []
        for offset in range(7):
            d = week_start + timedelta(days=offset)
            if d.year == year and d.month == month:
                days_in_week.append(d.day - 1)  # 0-indexed

        if days_in_week:
            label = (
                f"{week_start.strftime('%b %-d')} - "
                f"{week_end.strftime('%b %-d')}"
            )
            weeks.append({
                'start':        week_start,
                'end':          week_end,
                'label':        label,
                'days_in_month': days_in_week,
                'year':         year,
                'month':        month,
            })

        week_start += timedelta(days=7)

        if (week_start.year > year or
                (week_start.year == year and week_start.month > month)):
            break

    return weeks


# ─────────────────────────────────────────────────────────────────
# ELIGIBILITY HELPERS
# ─────────────────────────────────────────────────────────────────

def is_eligible(surgeon, role):
    """Check if surgeon is eligible for a given role."""
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


def is_active(surgeon, year, month):
    """
    Returns True if surgeon is active during the given month.
    - No start_date = active from beginning of time
    - start_date present = must have started by first day of this month
    - departure_date present = must not have departed before last day of month
    """
    last_day = monthrange(year, month)[1]
    month_start = datetime(year, month, 1)
    month_end = datetime(year, month, last_day)

    # Check start date
    start_str = surgeon.get('start_date', '')
    if start_str:
        try:
            sd = datetime.strptime(start_str[:10], '%Y-%m-%d')
            if sd > month_end:
                return False
        except Exception:
            pass

    # Check departure date
    depart_str = surgeon.get('departure_date', '')
    if depart_str:
        try:
            dd = datetime.strptime(depart_str[:10], '%Y-%m-%d')
            if dd < month_start:
                return False
        except Exception:
            pass

    return True


def is_fellow(surgeon):
    """Identify fellows by name convention."""
    return 'fellow' in surgeon.get('name', '').lower()


# ─────────────────────────────────────────────────────────────────
# FTE TARGET CALCULATION
# ─────────────────────────────────────────────────────────────────

def compute_block_target(surgeon, block_number, prior_totals, months):
    """
    Compute how many FTE service shifts this surgeon should serve
    across the full 6-month block.

    Block 1: Always starts at zero. Target = BLOCK_FTE_SHIFTS * fte
    Block 2: Target = annual_target - block1_actuals (what remains)

    Prorated if surgeon has start_date or departure_date within the block.
    """
    fte = float(surgeon.get('fte', 1.0))
    annual_target = ANNUAL_FTE_SHIFTS * fte

    if block_number == 1:
        block_target = BLOCK_FTE_SHIFTS * fte
    else:
        prior = float(prior_totals.get(surgeon.get('name', ''), 0))
        block_target = max(0.0, annual_target - prior)

    # Prorate for start_date within this block
    start_str = surgeon.get('start_date', '')
    if start_str:
        try:
            sd = datetime.strptime(start_str[:10], '%Y-%m-%d')
            block_start = datetime(months[0][0], months[0][1], 1)
            block_end_month = months[-1]
            last_day = monthrange(block_end_month[0], block_end_month[1])[1]
            block_end = datetime(block_end_month[0], block_end_month[1], last_day)

            if sd > block_start:
                total_days = (block_end - block_start).days + 1
                active_days = max(0, (block_end - sd).days + 1)
                block_target = block_target * (active_days / total_days)
        except Exception:
            pass

    # Prorate for departure_date within this block
    depart_str = surgeon.get('departure_date', '')
    if depart_str:
        try:
            dd = datetime.strptime(depart_str[:10], '%Y-%m-%d')
            block_start = datetime(months[0][0], months[0][1], 1)
            block_end_month = months[-1]
            last_day = monthrange(block_end_month[0], block_end_month[1])[1]
            block_end = datetime(block_end_month[0], block_end_month[1], last_day)

            if dd < block_end:
                total_days = (block_end - block_start).days + 1
                active_days = max(0, (dd - block_start).days + 1)
                block_target = block_target * (active_days / total_days)
        except Exception:
            pass

    return max(0.0, block_target)


# ─────────────────────────────────────────────────────────────────
# PREFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────

def get_surgeon_prefs(surgeon_id, preferences):
    """Find preferences for a specific surgeon."""
    for p in preferences:
        if p.get('surgeon_id') == surgeon_id:
            return p
    return {}


def parse_date_list(text):
    """
    Parse a free-text list of dates/ranges into a set of datetime.date objects.
    Handles formats like: 'Jan 5-12, Feb 20, Mar 18-22'
    """
    import re
    from datetime import date
    dates = set()
    if not text:
        return dates

    current_year = datetime.now().year
    months_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    # Split on commas
    parts = [p.strip() for p in text.split(',')]
    for part in parts:
        part = part.lower().strip()
        # Match "mon DD-DD" range
        range_match = re.match(
            r'([a-z]+)\s+(\d+)\s*[-–]\s*(\d+)', part
        )
        if range_match:
            mon = months_map.get(range_match.group(1)[:3])
            if mon:
                start_d = int(range_match.group(2))
                end_d = int(range_match.group(3))
                for day in range(start_d, end_d + 1):
                    try:
                        dates.add(date(current_year, mon, day))
                    except Exception:
                        pass
            continue
        # Match "mon DD"
        single_match = re.match(r'([a-z]+)\s+(\d+)', part)
        if single_match:
            mon = months_map.get(single_match.group(1)[:3])
            if mon:
                try:
                    dates.add(date(current_year, mon, int(single_match.group(2))))
                except Exception:
                    pass

    return dates


def week_overlaps_dates(week, date_set):
    """Check if any day in a week overlaps with a set of blocked dates."""
    from datetime import date
    for offset in range(7):
        d = week['start'] + timedelta(days=offset)
        if d.date() in date_set:
            return True
    return False


def day_in_dates(year, month, day_0indexed, date_set):
    """Check if a specific day (0-indexed) is in a set of blocked dates."""
    from datetime import date
    try:
        d = date(year, month, day_0indexed + 1)
        return d in date_set
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────
# FELLOW ROTATION HELPERS
# ─────────────────────────────────────────────────────────────────

def get_two_month_periods(months):
    """
    Split 6 months into three 2-month periods.
    Returns list of [(year,month), (year,month)] pairs.
    """
    periods = []
    for i in range(0, 6, 2):
        periods.append([months[i], months[i + 1]])
    return periods


# ─────────────────────────────────────────────────────────────────
# MAIN SOLVER — Full 6-month block
# ─────────────────────────────────────────────────────────────────

def solve_full_block(surgeons, months, block_number, preferences,
                     prior_totals, start_year):
    """
    Solve the entire 6-month block in a single OR-Tools model.
    months = list of (year, month) tuples, e.g. [(2026,7),(2026,8),...]
    """

    num_surgeons = len(surgeons)
    num_months = len(months)  # Always 6

    # ── Per-month week and day structure ──────────────────────────
    month_weeks = []   # month_weeks[m] = list of week dicts
    month_days = []    # month_days[m] = number of days in month

    for (y, mo) in months:
        weeks = get_weeks_for_month(y, mo)
        month_weeks.append(weeks)
        month_days.append(monthrange(y, mo)[1])

    # Total weeks across all months (for cross-month constraints)
    # We need a flat list of all weeks in order
    all_weeks = []
    week_to_month = []  # which month index each week belongs to
    for mi, weeks in enumerate(month_weeks):
        for w in weeks:
            all_weeks.append(w)
            week_to_month.append(mi)

    num_all_weeks = len(all_weeks)

    # ── Active surgeons per month ──────────────────────────────────
    # active_in_month[mi][s] = True/False
    active_in_month = []
    for mi, (y, mo) in enumerate(months):
        active = [is_active(surgeons[s], y, mo) for s in range(num_surgeons)]
        active_in_month.append(active)

    # ── Active surgeons per week (flat index) ─────────────────────
    active_in_week = []
    for wi, week in enumerate(all_weeks):
        mi = week_to_month[wi]
        active_in_week.append(active_in_month[mi])

    # ── Fellows ───────────────────────────────────────────────────
    fellow_indices = [
        s for s in range(num_surgeons)
        if is_fellow(surgeons[s])
    ]

    # ── Preference date sets per surgeon ──────────────────────────
    surgeon_time_off = {}
    surgeon_conferences = {}
    surgeon_avoid_nights = {}

    for s in range(num_surgeons):
        sid = surgeons[s].get('id', '')
        prefs = get_surgeon_prefs(sid, preferences)
        surgeon_time_off[s] = parse_date_list(prefs.get('time_off', ''))
        surgeon_conferences[s] = parse_date_list(prefs.get('conferences', ''))
        surgeon_avoid_nights[s] = parse_date_list(prefs.get('avoid_nights', ''))
        # Merge time_off and conferences for service week blocking
        surgeon_time_off[s] = surgeon_time_off[s] | surgeon_conferences[s]

    # ── Soft preference: call night preference ────────────────────
    # Read from surgeon profile — soft preference field
    # e.g. Rojas-Khalil has friday_saturday call preference
    surgeon_call_day_pref = {}
    for s in range(num_surgeons):
        pref = surgeons[s].get('call_day_preference', '')
        surgeon_call_day_pref[s] = pref  # e.g. 'friday_saturday', 'any', ''

    # ── Block FTE targets ─────────────────────────────────────────
    block_targets = [
        compute_block_target(surgeons[s], block_number, prior_totals, months)
        for s in range(num_surgeons)
    ]

    # ── OR-Tools Model ────────────────────────────────────────────
    model = cp_model.CpModel()

    # Weekly role variables [week_flat_index][surgeon_index]
    acs_msun = [
        [model.NewBoolVar(f'am_{wi}_{s}') for s in range(num_surgeons)]
        for wi in range(num_all_weeks)
    ]
    acs_mf = [
        [model.NewBoolVar(f'af_{wi}_{s}') for s in range(num_surgeons)]
        for wi in range(num_all_weeks)
    ]
    mcnair = [
        [model.NewBoolVar(f'mn_{wi}_{s}') for s in range(num_surgeons)]
        for wi in range(num_all_weeks)
    ]
    tsicu = [
        [model.NewBoolVar(f'ts_{wi}_{s}') for s in range(num_surgeons)]
        for wi in range(num_all_weeks)
    ]
    sicu = [
        [model.NewBoolVar(f'si_{wi}_{s}') for s in range(num_surgeons)]
        for wi in range(num_all_weeks)
    ]

    # Nightly call variables — indexed by [month_index][day_0indexed][surgeon]
    call = []
    for mi, (y, mo) in enumerate(months):
        days = month_days[mi]
        call_month = [
            [model.NewBoolVar(f'ca_{mi}_{d}_{s}') for s in range(num_surgeons)]
            for d in range(days)
        ]
        call.append(call_month)

    # ── HARD CONSTRAINT 1: Exactly one surgeon per weekly role ────
    for wi in range(num_all_weeks):
        model.AddExactlyOne(acs_msun[wi])
        model.AddExactlyOne(acs_mf[wi])
        model.AddExactlyOne(mcnair[wi])
        model.AddExactlyOne(tsicu[wi])
        model.AddExactlyOne(sicu[wi])

    # ── HARD CONSTRAINT 2: Exactly one call surgeon per night ─────
    for mi in range(num_months):
        for d in range(month_days[mi]):
            model.AddExactlyOne(call[mi][d])

    # ── HARD CONSTRAINT 3: Eligibility and active status ──────────
    for wi in range(num_all_weeks):
        mi = week_to_month[wi]
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
                if not active_in_month[mi][s] or not is_eligible(surgeons[s], 'call'):
                    model.Add(call[mi][d][s] == 0)

    # ── HARD CONSTRAINT 4: One role per surgeon per week ──────────
    for wi in range(num_all_weeks):
        for s in range(num_surgeons):
            model.Add(
                acs_msun[wi][s] + acs_mf[wi][s] + mcnair[wi][s] +
                tsicu[wi][s] + sicu[wi][s] <= 1
            )

    # ── HARD CONSTRAINT 5: No consecutive 7-day service weeks ─────
    # Applies across month boundaries using flat week index
    # ACS M-F (5 days) does NOT count as a 7-day week
    seven_day_roles = [acs_msun, mcnair, tsicu, sicu]
    for wi in range(num_all_weeks - 1):
        for s in range(num_surgeons):
            for r1 in seven_day_roles:
                for r2 in seven_day_roles:
                    model.Add(r1[wi][s] + r2[wi + 1][s] <= 1)

    # ── HARD CONSTRAINT 6: ACS M-Sun cannot repeat next week ──────
    for wi in range(num_all_weeks - 1):
        for s in range(num_surgeons):
            model.Add(acs_msun[wi][s] + acs_msun[wi + 1][s] <= 1)

    # ── HARD CONSTRAINT 7: ACS M-F and M-Sun different surgeons ───
    for wi in range(num_all_weeks):
        for s in range(num_surgeons):
            model.Add(acs_msun[wi][s] + acs_mf[wi][s] <= 1)

    # ── HARD CONSTRAINT 8: Call restrictions by weekly role ───────
    for wi, week in enumerate(all_weeks):
        mi = week_to_month[wi]
        y, mo = months[mi]
        week_start = week['start']

        for offset in range(7):
            day_dt = week_start + timedelta(days=offset)
            if day_dt.year != y or day_dt.month != mo:
                continue
            d = day_dt.day - 1   # 0-indexed
            dow = day_dt.weekday()  # 0=Mon, 6=Sun

            for s in range(num_surgeons):
                # McNair: NO call any night (24/7 commitment)
                model.Add(mcnair[wi][s] + call[mi][d][s] <= 1)

                # TSICU: no call Mon-Sat (dow 0-5)
                if dow <= 5:
                    model.Add(tsicu[wi][s] + call[mi][d][s] <= 1)

                # SICU: no call Mon-Sat
                if dow <= 5:
                    model.Add(sicu[wi][s] + call[mi][d][s] <= 1)

                # ACS M-Sun: no call Mon-Sat
                if dow <= 5:
                    model.Add(acs_msun[wi][s] + call[mi][d][s] <= 1)

                # ACS M-F: no call Mon-Thu (dow 0-3)
                if dow <= 3:
                    model.Add(acs_mf[wi][s] + call[mi][d][s] <= 1)

    # ── HARD CONSTRAINT 9: Max call nights per month per surgeon ──
    for mi in range(num_months):
        for s in range(num_surgeons):
            max_call = int(surgeons[s].get('max_call_per_month', 8))
            model.Add(
                sum(call[mi][d][s] for d in range(month_days[mi])) <= max_call
            )

    # ── HARD CONSTRAINT 10: Fellows cannot share same role ────────
    if len(fellow_indices) >= 2:
        for wi in range(num_all_weeks):
            for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                model.Add(
                    sum(role[wi][f] for f in fellow_indices) <= 1
                )

    # ── HARD CONSTRAINT 11: Fellow rotation ──────────────────────
    # Each active fellow: exactly 2 ACS weeks + 1 SICU week
    # per 2-month period (3 periods in 6-month block)
    two_month_periods = get_two_month_periods(months)

    for period_idx, period_months in enumerate(two_month_periods):
        # Get flat week indices that belong to this 2-month period
        period_week_indices = [
            wi for wi, week in enumerate(all_weeks)
            if (week['year'], week['month']) in
               [(pm[0], pm[1]) for pm in period_months]
        ]

        for f in fellow_indices:
            # Check if fellow is active in this period
            period_active = any(
                active_in_month[
                    months.index(pm) if pm in months else 0
                ][f]
                for pm in period_months
                if pm in months
            )

            if not period_active:
                continue

            # Exactly 2 ACS weeks (M-F + M-Sun combined) per period
            acs_in_period = [
                acs_msun[wi][f] + acs_mf[wi][f]
                for wi in period_week_indices
            ]
            if acs_in_period:
                model.Add(sum(acs_in_period) == 2)

            # Exactly 1 SICU week per period
            sicu_in_period = [sicu[wi][f] for wi in period_week_indices]
            if sicu_in_period:
                model.Add(sum(sicu_in_period) == 1)

    # ── HARD CONSTRAINT 12: Time off / conference blocking ────────
    for wi, week in enumerate(all_weeks):
        mi = week_to_month[wi]
        for s in range(num_surgeons):
            blocked = surgeon_time_off[s]
            if blocked and week_overlaps_dates(week, blocked):
                model.Add(acs_msun[wi][s] == 0)
                model.Add(acs_mf[wi][s] == 0)
                model.Add(mcnair[wi][s] == 0)
                model.Add(tsicu[wi][s] == 0)
                model.Add(sicu[wi][s] == 0)

    # Block call on avoided nights
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for s in range(num_surgeons):
                avoid = surgeon_avoid_nights[s]
                if avoid and day_in_dates(y, mo, d, avoid):
                    model.Add(call[mi][d][s] == 0)

    # ── HARD CONSTRAINT 13: Max 14 service shifts per month ───────
    # Hard for baseline surgeons, soft for willing/seeking (handled in objective)
    for mi in range(num_months):
        # Get weeks belonging to this month
        month_wi = [
            wi for wi in range(num_all_weeks)
            if week_to_month[wi] == mi
        ]
        for s in range(num_surgeons):
            pref = surgeons[s].get('extra_shift_preference', 'baseline')
            if pref == 'baseline':
                # Hard cap at 14 shifts
                shifts_this_month = []
                for wi in month_wi:
                    shifts_this_month.append(5 * acs_mf[wi][s])
                    shifts_this_month.append(7 * acs_msun[wi][s])
                    shifts_this_month.append(7 * mcnair[wi][s])
                    shifts_this_month.append(7 * tsicu[wi][s])
                    shifts_this_month.append(7 * sicu[wi][s])
                if shifts_this_month:
                    model.Add(sum(shifts_this_month) <= MAX_SERVICE_SHIFTS_PER_MONTH)

    # ── SOFT OBJECTIVE ────────────────────────────────────────────
    obj_terms = []      # things to maximize
    penalty_terms = []  # things to minimize

    # ── FTE equity: minimize squared deviation from target ────────
    # We approximate this by rewarding surgeons who are below target
    # and penalizing those above it, weighted by their preference
    for s in range(num_surgeons):
        target = block_targets[s]
        pref = surgeons[s].get('extra_shift_preference', 'baseline')
        weight = max(1, int(target))

        # Reward assignments proportionally to close the gap to target
        for wi in range(num_all_weeks):
            if active_in_week[wi][s]:
                if is_eligible(surgeons[s], 'acs_mf'):
                    obj_terms.append(weight * acs_mf[wi][s])
                if is_eligible(surgeons[s], 'acs_msun'):
                    obj_terms.append(weight * acs_msun[wi][s])
                if is_eligible(surgeons[s], 'mcnair'):
                    obj_terms.append(weight * mcnair[wi][s])
                if is_eligible(surgeons[s], 'tsicu'):
                    obj_terms.append(weight * tsicu[wi][s])
                if is_eligible(surgeons[s], 'sicu'):
                    obj_terms.append(weight * sicu[wi][s])

        # Penalize over-assignment for baseline surgeons
        if pref == 'baseline':
            for wi in range(num_all_weeks):
                if active_in_week[wi][s]:
                    for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                        penalty_terms.append(role[wi][s])

    # ── Weekend call: soft minimize, prefer willing/seeking ───────
    for mi, (y, mo) in enumerate(months):
        weekend_days_mi = [
            d for d in range(month_days[mi])
            if datetime(y, mo, d + 1).weekday() >= 4  # Fri=4, Sat=5, Sun=6
        ]
        for d in weekend_days_mi:
            for s in range(num_surgeons):
                pref = surgeons[s].get('extra_shift_preference', 'baseline')
                if pref == 'baseline':
                    # Penalize weekend call for baseline surgeons
                    penalty_terms.append(3 * call[mi][d][s])
                elif pref == 'willing':
                    # Slight reward for willing surgeons taking weekend call
                    obj_terms.append(call[mi][d][s])
                elif pref == 'seeking':
                    # Stronger reward for seeking surgeons taking weekend call
                    obj_terms.append(2 * call[mi][d][s])

    # ── Call day preferences (from surgeon profile) ───────────────
    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            dow = datetime(y, mo, d + 1).weekday()
            for s in range(num_surgeons):
                pref_days = surgeon_call_day_pref[s]
                if pref_days == 'friday_saturday':
                    # Penalize non-Fri/Sat call
                    if dow not in (4, 5):
                        penalty_terms.append(2 * call[mi][d][s])
                    else:
                        obj_terms.append(call[mi][d][s])

    # ── ACS/ICU allocation soft targets ───────────────────────────
    for s in range(num_surgeons):
        acs_alloc = float(surgeons[s].get('acs_allocation', 0.5))
        icu_alloc = float(surgeons[s].get('icu_allocation', 0.5))
        target = block_targets[s]
        acs_target = int(target * acs_alloc)
        icu_target = int(target * icu_alloc)

        # Reward getting close to ACS target
        acs_assignments = [
            acs_mf[wi][s] + acs_msun[wi][s]
            for wi in range(num_all_weeks)
            if active_in_week[wi][s]
               and is_eligible(surgeons[s], 'acs_mf')
        ]

        # Reward getting close to ICU target
        icu_assignments = [
            mcnair[wi][s] + tsicu[wi][s] + sicu[wi][s]
            for wi in range(num_all_weeks)
            if active_in_week[wi][s]
               and (is_eligible(surgeons[s], 'mcnair')
                    or is_eligible(surgeons[s], 'tsicu')
                    or is_eligible(surgeons[s], 'sicu'))
        ]

    # ── Build and set objective ───────────────────────────────────
    if obj_terms or penalty_terms:
        objective = []
        if obj_terms:
            objective.append(sum(obj_terms))
        if penalty_terms:
            objective.append(-sum(penalty_terms))
        model.Maximize(sum(objective) if len(objective) > 1 else objective[0])

    # ── SOLVE ─────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0   # More time for full block
    solver.parameters.num_search_workers = 4

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        # Stage 2: Relax baseline monthly cap and retry
        raise Exception(
            f"No valid schedule found for block. "
            f"Status: {solver.StatusName(status)}. "
            f"Check surgeon eligibility and time-off conflicts."
        )

    # ── BUILD OUTPUT ──────────────────────────────────────────────
    result = {}

    for mi, (y, mo) in enumerate(months):
        mk = f"{y}-{str(mo).zfill(2)}"
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
                        'Call': surgeons[s]['name'],
                        'Backup': ''
                    }

        # FTE summary for this month
        fte_summary = {}
        for s in range(num_surgeons):
            name = surgeons[s]['name']
            shifts = 0
            for w in result_weeks:
                if w.get('ACS (M-F)') == name:
                    shifts += SHIFTS_ACS_MF
                if w.get('ACS (M-Sun)') == name:
                    shifts += SHIFTS_ACS_MSUN
                for role in ['McNair ICU', 'TSICU', 'SICU']:
                    if w.get(role) == name:
                        shifts += SHIFTS_ICU
            fte_summary[name] = shifts

        result[mk] = {
            'weeks':  result_weeks,
            'nights': result_nights,
            'fte_summary': fte_summary,
        }

    # ── VALIDATION ────────────────────────────────────────────────
    violations = []
    warnings = []

    for mi, (y, mo) in enumerate(months):
        mk = f"{y}-{str(mo).zfill(2)}"
        month_data = result[mk]
        month_label = datetime(y, mo, 1).strftime('%B %Y')

        # All weekly roles filled
        for w in month_data['weeks']:
            for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                if role not in w:
                    violations.append(
                        f"{month_label} week {w['label']}: {role} not assigned"
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
            for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                name = w.get(role)
                if name:
                    if name in seen:
                        violations.append(
                            f"{month_label} {w['label']}: "
                            f"{name} in both {seen[name]} and {role}"
                        )
                    seen[name] = role

        # No inactive surgeon assigned
        for w in month_data['weeks']:
            for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
                name = w.get(role, '')
                for s in range(num_surgeons):
                    if surgeons[s]['name'] == name:
                        if not active_in_month[mi][s]:
                            violations.append(
                                f"{month_label} {w['label']}: "
                                f"{name} assigned but not active"
                            )

        # Eligibility check
        for w in month_data['weeks']:
            checks = [
                ('ACS (M-Sun)', 'acs_msun'),
                ('ACS (M-F)',   'acs_mf'),
                ('McNair ICU',  'mcnair'),
                ('TSICU',       'tsicu'),
                ('SICU',        'sicu'),
            ]
            for role_label, role_key in checks:
                name = w.get(role_label, '')
                if name:
                    for s in range(num_surgeons):
                        if surgeons[s]['name'] == name:
                            if not is_eligible(surgeons[s], role_key):
                                violations.append(
                                    f"{month_label} {w['label']}: "
                                    f"{name} not eligible for {role_label}"
                                )

    # Consecutive 7-day week check across full block
    seven_day_role_labels = ['ACS (M-Sun)', 'McNair ICU', 'TSICU', 'SICU']
    flat_weeks_output = []
    for mi in range(num_months):
        mk = f"{months[mi][0]}-{str(months[mi][1]).zfill(2)}"
        for w in result[mk]['weeks']:
            flat_weeks_output.append((months[mi], w))

    for i in range(len(flat_weeks_output) - 1):
        w1 = flat_weeks_output[i][1]
        w2 = flat_weeks_output[i + 1][1]
        for role1 in seven_day_role_labels:
            for role2 in seven_day_role_labels:
                name1 = w1.get(role1)
                name2 = w2.get(role2)
                if name1 and name1 == name2:
                    violations.append(
                        f"Consecutive 7-day weeks: {name1} "
                        f"({role1} then {role2})"
                    )

    # Fellow rotation check
    for period_idx, period_months in enumerate(two_month_periods):
        for f in fellow_indices:
            fname = surgeons[f]['name']
            acs_count = 0
            sicu_count = 0
            period_active = False

            for pm in period_months:
                if pm not in months:
                    continue
                mi = months.index(pm)
                if not active_in_month[mi][f]:
                    continue
                period_active = True
                mk = f"{pm[0]}-{str(pm[1]).zfill(2)}"
                for w in result[mk]['weeks']:
                    if w.get('ACS (M-F)') == fname:
                        acs_count += 1
                    if w.get('ACS (M-Sun)') == fname:
                        acs_count += 1
                    if w.get('SICU') == fname:
                        sicu_count += 1

            if period_active:
                if acs_count != 2:
                    violations.append(
                        f"Fellow {fname} period {period_idx + 1}: "
                        f"{acs_count} ACS weeks (expected 2)"
                    )
                if sicu_count != 1:
                    violations.append(
                        f"Fellow {fname} period {period_idx + 1}: "
                        f"{sicu_count} SICU weeks (expected 1)"
                    )

    # FTE summary across full block
    block_fte_summary = {}
    for s in range(num_surgeons):
        name = surgeons[s]['name']
        total = sum(
            result[f"{y}-{str(mo).zfill(2)}"]['fte_summary'].get(name, 0)
            for y, mo in months
        )
        block_fte_summary[name] = {
            'served':  total,
            'target':  round(block_targets[s], 1),
            'delta':   round(total - block_targets[s], 1),
        }

    return {
        'months':     result,
        'validation': {
            'violations':        violations,
            'warnings':          warnings,
            'valid':             len(violations) == 0,
            'block_fte_summary': block_fte_summary,
        }
    }


# ─────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
