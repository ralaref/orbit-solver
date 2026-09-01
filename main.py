"""
ORbit Surgical Scheduling Solver v27
=====================================
v27: PACE SCARCITY for elective surgeons.

v26 gave a surgeon two protected weeks a month and honoured them, but left them
short of target — 75 against 84 on the first Block 2 run, with four available
weeks sitting unworked. The weeks were there; the greedy never routed them.

The cause is in pace_score. Deficit is measured as a FRACTION OF TARGET:

    pace_deficit = (budget - served) / target

so the same absolute shortfall reads differently depending on the size of the
target. Seven shifts behind is 0.083 on a target of 84 and 0.167 on a target of
42. A 1.0 FTE surgeon therefore loses ties to half-timers who look twice as far
behind while being the same distance behind in real terms.

That is a general property of the scoring and changing it moves everybody, so
v27 does not touch it. What it does is correct for the thing that makes it bite
an elective surgeon specifically: they have fourteen chances to reach the same
number, not twenty-six. Each available week carries nearly twice the weight, and
a week they are passed over for cannot be made up later in the way it can for
someone with the whole block available.

    scarcity = active weeks / available weeks   (capped at 2.0)

applied to the deficit only when they are BEHIND. Ahead of pace they yield
normally — an elective practice is a reason to be routed efficiently into the
weeks you have, not a reason to be routed past your target.

Surgeons with no elective practice have a scarcity of exactly 1.0 and are scored
byte for byte as in v26.

v26: ELECTIVE PRACTICE. Two protected weeks a month — an elective week (off
service, a fixed number of call nights inside it) and an admin week (off
service, no call at all). Everything soft; coverage still wins over preference.
Consecutive 7-day weeks became a per-surgeon ceiling rather than a flat bar.

The weekend-spacing defect is NOT addressed here and remains open: solve_call
balances weekend COUNTS with no concept of spacing, so twelve weekends spread
evenly and twelve stacked six-deep still score identically.

v25: The boundary rule now applies to CHECKING as well as building.
v24: BLOCK BOUNDARY. A week belongs to the month containing its Monday.
v23: CHECK-ONLY MODE. All flag/violation/warning logic in validate_schedule().
v22: Observability + preference-honoring hardening.
v21: Ranked time-off week preferences as soft penalties in pace_score.
v20: Block target = 84 x FTE for every block; over-target is compensation.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from calendar import monthrange
import os

app = Flask(__name__)
CORS(app)

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

ROLE_ORDER = ['SICU', 'TSICU', 'McNair ICU', 'ACS (M-Sun)', 'ACS (M-F)']

# ── Call-avoidance penalty tiers for requested weeks off (v22) ────────────────
CALL_OFFWEEK_PENALTY = {'top': 500, 'mid': 250, 'low': 80}
CALL_OFFWEEK_HOLIDAY_MULT = 1.5

# ── Elective practice (v26) ──────────────────────────────────────────────────
# Sized against the terms already in the objective: weekend fairness is 40/20,
# a four-night run is 25, a call-day preference is 3. A reward of 150 reliably
# pulls the night to the person who asked for it without swamping the rest.
#
# The avoid penalty is deliberately below the rank-1 off-week penalty of 500:
# not wanting call the night before an elective week is a preference, and it
# should lose to a night nobody else can cover.
ELECTIVE_CALL_REWARD    = 150
ELECTIVE_AVOID_PENALTY  = 200

# ── Pace scarcity (v27) ──────────────────────────────────────────────────────
# The ceiling on how much an elective surgeon's deficit may be amplified. Two
# protected weeks a month gives 26/14 = 1.86, so the cap does not bind today —
# it exists so that a future arrangement with very few available weeks cannot
# make one person outscore the entire division on every remaining week.
PACE_SCARCITY_CAP = 2.0

# Python's weekday(): Monday is 0.
DOW_KEYS = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6}


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ORbit Solver v27'})


@app.route('/solve-block', methods=['POST'])
def solve_block():
    try:
        data            = request.json
        surgeons        = data.get('surgeons', [])
        block_number    = data.get('block_number', 1)
        start_year      = data.get('start_year')
        preferences     = data.get('preferences', [])
        prior_totals    = data.get('prior_totals', {})
        holiday_history = data.get('holiday_history', {}) or {}

        if not surgeons:
            return jsonify({'success': False, 'error': 'No surgeons provided'}), 400
        if not start_year:
            return jsonify({'success': False, 'error': 'start_year required'}), 400

        if block_number == 1:
            months = [(start_year, m) for m in BLOCK1_MONTHS]
        else:
            months = [(start_year + 1, m) for m in BLOCK2_MONTHS]

        block_start = get_block_start(months)

        for i, s in enumerate(surgeons):
            s['_idx'] = i

        print("=== v27 SOLVER STARTED ===", flush=True)
        print(f"DEBUG block_number={block_number} start_year={start_year} months={months}", flush=True)
        print(f"DEBUG block_start={block_start.strftime('%Y-%m-%d') if block_start else None}", flush=True)
        for s in surgeons:
            print(f"  {s['name']} | is_fellow={s.get('is_fellow')} | fte={s.get('fte')} | sicu={s.get('covers_sicu')} | acs={s.get('can_acs')}", flush=True)

        for s in surgeons:
            el = s.get('elective') or {}
            if el:
                print(f"DEBUG elective {s['name']}: "
                      f"elective_weeks={el.get('elective_weeks')} "
                      f"admin_weeks={el.get('admin_weeks')} "
                      f"call_days={el.get('call_days')} "
                      f"sun_before_admin={el.get('call_sunday_before_admin')} "
                      f"seven_day_only={el.get('seven_day_weeks_only')} "
                      f"max_consec={el.get('max_consecutive_service_weeks')}", flush=True)

        week_assignments = greedy_service_weeks(
            surgeons=surgeons,
            months=months,
            block_number=block_number,
            preferences=preferences,
            prior_totals=prior_totals,
            holiday_history=holiday_history,
            block_start=block_start,
        )

        call_assignments = solve_call(
            surgeons=surgeons,
            months=months,
            week_assignments=week_assignments,
            preferences=preferences,
            block_start=block_start,
        )

        result = build_output(
            surgeons=surgeons,
            months=months,
            week_assignments=week_assignments,
            call_assignments=call_assignments,
            block_number=block_number,
            prior_totals=prior_totals,
            preferences=preferences,
            holiday_history=holiday_history,
            block_start=block_start,
        )

        return jsonify({'success': True, 'schedule': result})

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


@app.route('/validate-only', methods=['POST'])
def validate_only():
    """
    Fast check-only mode. Runs validate_schedule() against a schedule the app
    sends (already assembled: months -> weeks/nights), with NO optimizing.
    Returns the same 'validation' shape as a full solve, in well under a second.
    """
    try:
        data            = request.json or {}
        schedule        = data.get('schedule', {})     # {month_key: {weeks, nights}}
        surgeons        = data.get('surgeons', [])
        block_number    = data.get('block_number', 1)
        start_year      = data.get('start_year')
        preferences     = data.get('preferences', [])
        prior_totals    = data.get('prior_totals', {})
        holiday_history = data.get('holiday_history', {}) or {}

        if not schedule:
            return jsonify({'success': False, 'error': 'No schedule provided'}), 400
        if not surgeons:
            return jsonify({'success': False, 'error': 'No surgeons provided'}), 400
        if not start_year:
            return jsonify({'success': False, 'error': 'start_year required'}), 400

        if block_number == 1:
            months = [(start_year, m) for m in BLOCK1_MONTHS]
        else:
            months = [(start_year + 1, m) for m in BLOCK2_MONTHS]

        block_start = get_block_start(months)

        validation = validate_schedule(
            schedule, surgeons, months, block_number, prior_totals,
            preferences=preferences, holiday_history=holiday_history,
            block_start=block_start)

        return jsonify({'success': True, 'validation': validation})

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


# ─────────────────────────────────────────────────────────────────
# BLOCK BOUNDARY (v24)
# ─────────────────────────────────────────────────────────────────

def get_block_start(months):
    """
    The Monday of the first week this block owns.

    A week belongs to the month containing its Monday. So a block begins at the
    first Monday that falls inside its first month; anything earlier sits in a
    week the previous month already owns and has already been published.

    Computed from the calendar every time, never hardcoded:
      Jan 2027 starts Friday  -> Jan 4, 2027
      Jul 2027 starts Thursday-> Jul 5, 2027
      Jan 2028 starts Saturday-> Jan 3, 2028
      a month starting Monday -> the 1st
    """
    if not months:
        return None
    y, mo     = months[0]
    first_day = datetime(y, mo, 1)
    if first_day.weekday() == 0:
        return first_day
    return first_day + timedelta(days=7 - first_day.weekday())


def block_day_offsets(months, block_start):
    """
    0-indexed first day to include, per month.

    Only the block's first month can carry leading days that belong to the
    previous block's last week. Every other month starts at day 0.
    """
    offsets = [0] * len(months)
    if block_start is not None and months:
        y, mo = months[0]
        if block_start.year == y and block_start.month == mo:
            offsets[0] = block_start.day - 1
    return offsets


# ─────────────────────────────────────────────────────────────────
# WEEK UTILITIES
# ─────────────────────────────────────────────────────────────────

def get_all_weeks(months, block_start=None):
    seen  = set()
    weeks = []
    for mi, (y, mo) in enumerate(months):
        first_day  = datetime(y, mo, 1)
        week_start = first_day - timedelta(days=first_day.weekday())
        while True:
            # v24: a week starting before the block's first Monday belongs to
            # the previous month, which already owns and has published it.
            if block_start is not None and week_start < block_start:
                week_start += timedelta(days=7)
                if week_start.year > y or (week_start.year == y and week_start.month > mo):
                    break
                continue
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
    # For a real block this is exactly [(0,1),(2,3),(4,5)]. Guard against
    # shorter month lists so check-only never index-errors on partial input.
    return [[months[i], months[i + 1]] for i in range(0, len(months) - 1, 2)]


# ─────────────────────────────────────────────────────────────────
# SURGEON HELPERS
# ─────────────────────────────────────────────────────────────────

def is_eligible(surgeon, role):
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
    if surgeon.get('is_fellow') is True:
        return True
    has_start  = bool(surgeon.get('start_date'))
    half_fte   = abs(float(surgeon.get('fte', 1.0)) - 0.5) < 0.01
    can_sicu   = bool(surgeon.get('covers_sicu'))
    can_acs    = bool(surgeon.get('can_acs'))
    no_mcnair  = not bool(surgeon.get('covers_mcnair'))
    no_tsicu   = not bool(surgeon.get('covers_tsicu'))
    return has_start and half_fte and can_sicu and can_acs and no_mcnair and no_tsicu


def get_pref(surgeon):
    return surgeon.get('extra_shift_preference', 'baseline') or 'baseline'


def is_seven_day_role(role):
    return role in ('ACS (M-Sun)', 'McNair ICU', 'TSICU', 'SICU')


# ─────────────────────────────────────────────────────────────────
# ELECTIVE PRACTICE (v26)
# ─────────────────────────────────────────────────────────────────

def get_elective(surgeon):
    """The elective block off a surgeon payload, or an empty dict."""
    el = surgeon.get('elective')
    return el if isinstance(el, dict) else {}


def elective_max_consecutive(surgeon):
    """
    How many 7-day service weeks this surgeon may work back to back.

    One for everybody by default, which is the rule as it has always stood. A
    surgeon who has opted in can take two — never three, because the ceiling is
    explicit rather than the constraint simply being removed. Left as a number
    so a third is a data change and a deliberate one.
    """
    el = get_elective(surgeon)
    try:
        n = int(el.get('max_consecutive_service_weeks', 1))
    except Exception:
        n = 1
    return 2 if n >= 2 else 1


def elective_prefers_seven_day(surgeon):
    return bool(get_elective(surgeon).get('seven_day_weeks_only'))


def elective_protected_labels(surgeon):
    """Every week label this surgeon holds off service by arrangement."""
    el = get_elective(surgeon)
    if not el:
        return set()
    return set(el.get('elective_weeks') or []) | set(el.get('admin_weeks') or [])


def date_to_slot(dt, months, day_start):
    """A date -> (month index, 0-indexed day), or None if outside the block."""
    for mi, (y, mo) in enumerate(months):
        if dt.year == y and dt.month == mo:
            d = dt.day - 1
            return (mi, d) if d >= day_start[mi] else None
    return None


def build_elective_plan(surgeons, months, week_assignments, day_start):
    """
    surgeon index -> {'call': set of (mi, d), 'avoid': set of (mi, d)}

    'call' is every night the surgeon asked for: the chosen days inside each
    elective week, plus the Sunday before each admin week when that is switched
    on. 'avoid' is the Sunday immediately before each elective week — walking
    into the week post-call is the thing the week exists to prevent.

    Weeks are matched by label against the weeks the greedy actually built, so
    a label that matches nothing contributes nothing rather than silently
    shifting dates. Admin weeks themselves are absent from both sets: the
    rank-1 time-off entry the route sends already keeps call out of them, which
    is exactly the wanted behaviour.
    """
    label_start = {}
    for wa in week_assignments.values():
        if wa.get('label'):
            label_start[wa['label']] = wa['start']

    plan = {}
    for i, s in enumerate(surgeons):
        el = get_elective(s)
        if not el:
            continue

        elective_weeks = el.get('elective_weeks') or []
        admin_weeks    = el.get('admin_weeks') or []
        day_offsets    = [DOW_KEYS[d] for d in (el.get('call_days') or [])
                          if d in DOW_KEYS]
        want_sunday    = bool(el.get('call_sunday_before_admin'))

        call_slots = set()
        avoid_slots = set()
        unmatched = []

        for lbl in elective_weeks:
            start = label_start.get(lbl)
            if start is None:
                unmatched.append(lbl)
                continue
            for off in day_offsets:
                slot = date_to_slot(start + timedelta(days=off), months, day_start)
                if slot:
                    call_slots.add(slot)
            slot = date_to_slot(start - timedelta(days=1), months, day_start)
            if slot:
                avoid_slots.add(slot)

        for lbl in admin_weeks:
            start = label_start.get(lbl)
            if start is None:
                unmatched.append(lbl)
                continue
            if want_sunday:
                slot = date_to_slot(start - timedelta(days=1), months, day_start)
                if slot:
                    call_slots.add(slot)

        if unmatched:
            print(f"DEBUG elective {s['name']}: labels matched no week: {unmatched}",
                  flush=True)

        # A night cannot be both asked for and avoided. The ask wins — it is the
        # more specific instruction. This only fires if an elective week is
        # scheduled directly after an admin week and the Sunday is shared.
        avoid_slots -= call_slots

        if call_slots or avoid_slots:
            plan[i] = {'call': call_slots, 'avoid': avoid_slots}
            print(f"DEBUG elective {s['name']}: {len(call_slots)} call night(s) "
                  f"requested, {len(avoid_slots)} avoided", flush=True)

    return plan


def compute_block_target(surgeon, block_number, prior_totals, months):
    fte          = float(surgeon.get('fte', 1.0))
    block_target = BLOCK_FTE_SHIFTS * fte

    block_start = datetime(months[0][0], months[0][1], 1)
    last        = monthrange(months[-1][0], months[-1][1])[1]
    block_end   = datetime(months[-1][0], months[-1][1], last)
    total_days  = (block_end - block_start).days + 1

    start_str = surgeon.get('start_date') or ''
    if start_str:
        try:
            sd = datetime.strptime(start_str[:10], '%Y-%m-%d')
            if sd > block_end:
                return 0.0
            if sd > block_start:
                active       = max(0, (block_end - sd).days + 1)
                block_target = block_target * (active / total_days)
        except Exception:
            pass

    depart_str = surgeon.get('departure_date') or ''
    if depart_str:
        try:
            dd = datetime.strptime(depart_str[:10], '%Y-%m-%d')
            if dd < block_start:
                return 0.0
            if dd < block_end:
                active       = max(0, (dd - block_start).days + 1)
                block_target = block_target * (active / total_days)
        except Exception:
            pass

    return max(0.0, block_target)


def compute_soft_cap(target_shifts, pref):
    multiplier = {'baseline': 1.0, 'willing': 1.4, 'seeking': 1.8}.get(pref, 1.0)
    return round(target_shifts * multiplier)


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
# RANKED WEEK PREFERENCE PARSING (shared by all stages)
# ─────────────────────────────────────────────────────────────────

def build_week_ranks(surgeons, preferences):
    """surgeon_name -> { week_label: {'rank': int, 'is_holiday': bool} }"""
    out = {}
    for s in surgeons:
        prefs = get_surgeon_prefs(s.get('id', ''), preferences)
        tow   = prefs.get('time_off_weeks', [])
        d     = {}
        if isinstance(tow, list):
            for w in tow:
                lbl = w.get('week', '')
                if lbl:
                    d[lbl] = {
                        'rank':       int(w.get('rank', 99)),
                        'is_holiday': bool(w.get('isHoliday', False)),
                        'source':     w.get('source', ''),
                    }
        out[s['name']] = d
    return out


def holiday_claim_factor(name, holiday_history):
    if not holiday_history:
        return 1.0
    info = holiday_history.get(name)
    if not info:
        return 1.0
    try:
        worked = int(info.get('recent_holidays_worked', 0))
    except Exception:
        worked = 0
    return 1.0 + min(worked * 0.25, 1.0)


def compute_fellow_period_targets(fellows, periods, all_weeks):
    result = {}
    for fellow in fellows:
        fname         = fellow['name']
        result[fname] = {}
        for pi, period_months in enumerate(periods):
            period_set   = set((pm[0], pm[1]) for pm in period_months)
            period_weeks = [w for w in all_weeks
                            if (w['year'], w['month']) in period_set]
            total  = len(period_weeks)
            active = sum(1 for w in period_weeks if is_active_for_week(fellow, w))
            if total == 0 or active == 0:
                result[fname][pi] = (0, 0)
                continue
            ratio             = active / total
            result[fname][pi] = (max(0, round(2 * ratio)),
                                  max(0, round(1 * ratio)))
    return result


def greedy_service_weeks(surgeons, months, block_number, preferences,
                         prior_totals, holiday_history=None, block_start=None):
    holiday_history = holiday_history or {}
    all_weeks = get_all_weeks(months, block_start)
    periods   = get_two_month_periods(months)

    fellows   = [s for s in surgeons if is_fellow(s)]
    all_names = [s['name'] for s in surgeons]

    print(f"DEBUG fellows detected: {[f['name'] for f in fellows]}", flush=True)
    print(f"DEBUG total weeks generated: {len(all_weeks)}", flush=True)
    if all_weeks:
        print(f"DEBUG first week: {all_weeks[0]['label']} | last week: {all_weeks[-1]['label']}", flush=True)

    surgeon_time_off = {}
    for s in surgeons:
        prefs = get_surgeon_prefs(s.get('id', ''), preferences)
        y_ref = months[0][0]
        off   = parse_date_list(prefs.get('time_off',    ''), y_ref)
        conf  = parse_date_list(prefs.get('conferences', ''), y_ref)
        surgeon_time_off[s['name']] = off | conf

    surgeon_week_ranks = build_week_ranks(surgeons, preferences)
    for name, d in surgeon_week_ranks.items():
        if d:
            print(f"DEBUG {name} time_off_weeks: {list(d.keys())}", flush=True)

    # v26: per-surgeon ceiling on consecutive 7-day weeks, and whether a 5-day
    # ACS week should be discouraged. Both default to the v25 behaviour.
    max_consec_7day = {s['name']: elective_max_consecutive(s) for s in surgeons}
    prefers_7day    = {s['name']: elective_prefers_seven_day(s) for s in surgeons}
    for name, n in max_consec_7day.items():
        if n > 1:
            print(f"DEBUG {name}: may work up to {n} consecutive 7-day weeks", flush=True)

    targets   = {}
    soft_caps = {}
    for s in surgeons:
        t                    = compute_block_target(s, block_number, prior_totals, months)
        t_int                = max(0, round(t))
        pref                 = get_pref(s)
        targets[s['name']]   = t_int
        soft_caps[s['name']] = compute_soft_cap(t_int, pref)

    print("DEBUG targets:", {k: v for k, v in targets.items()}, flush=True)
    print("DEBUG soft_caps:", {k: v for k, v in soft_caps.items()}, flush=True)

    surgeon_active_weeks = {
        s['name']: sum(1 for w in all_weeks if is_active_for_week(s, w))
        for s in surgeons
    }

    # ── v27: pace scarcity ────────────────────────────────────────────────
    # An elective surgeon carries the same target across far fewer weeks, and
    # a week they lose cannot be made up the way it can for someone with the
    # whole block open. Their deficit is scaled by how scarce their available
    # weeks are, so a tie against someone with twice the runway goes to them.
    #
    # Every other surgeon gets exactly 1.0 and is scored as in v26.
    week_label_set = {w['label'] for w in all_weeks}
    pace_scarcity  = {}
    for s in surgeons:
        name      = s['name']
        protected = elective_protected_labels(s) & week_label_set
        active    = surgeon_active_weeks[name]
        available = active - len(protected)
        if not protected or available <= 0 or active <= 0:
            pace_scarcity[name] = 1.0
            continue
        pace_scarcity[name] = min(active / available, PACE_SCARCITY_CAP)
        print(f"DEBUG {name}: {available} of {active} weeks available, "
              f"pace scarcity {pace_scarcity[name]:.2f}", flush=True)

    served           = {n: 0   for n in all_names}
    last_service_wi  = {n: -99 for n in all_names}
    last_7day_wi     = {n: -99 for n in all_names}
    last_acs_msun_wi = {n: -99 for n in all_names}
    active_so_far    = {n: 0   for n in all_names}
    # How many 7-day weeks the surgeon has worked back to back, counting the
    # run that ends at last_7day_wi. Reset whenever there is a gap.
    run_7day         = {n: 0   for n in all_names}

    fellow_period_targets = compute_fellow_period_targets(fellows, periods, all_weeks)
    fellow_acs_served  = {f['name']: {pi: 0 for pi in range(len(periods))} for f in fellows}
    fellow_sicu_served = {f['name']: {pi: 0 for pi in range(len(periods))} for f in fellows}

    def get_period_idx(week):
        for pi, period_months in enumerate(periods):
            if (week['year'], week['month']) in set((pm[0], pm[1]) for pm in period_months):
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

    def base_eligible(surgeon, role, week, wi, assigned_this_week):
        name = surgeon['name']
        if name in assigned_this_week.values():
            return False
        if not is_active_for_week(surgeon, week):
            return False
        if not is_eligible(surgeon, role):
            return False
        if surgeon_time_off.get(name) and week_overlaps_dates(
                week, surgeon_time_off[name]):
            return False
        # v26: was a flat bar on two 7-day weeks in a row. Now a ceiling, which
        # is the same bar for everyone whose ceiling is one — the default.
        if is_seven_day_role(role):
            gap = wi - last_7day_wi[name]
            if gap <= 1:
                if not (gap == 1 and run_7day[name] < max_consec_7day[name]):
                    return False
        if role == 'ACS (M-Sun)' and wi - last_acs_msun_wi[name] <= 1:
            return False
        if is_fellow(surgeon) and not fellow_can_take_role(surgeon, role, week):
            return False
        return True

    def within_cap(surgeon):
        name = surgeon['name']
        return served[name] < soft_caps[name]

    def pace_score(surgeon, wi, role):
        name  = surgeon['name']
        t     = targets[name]
        pref  = get_pref(surgeon)
        if t == 0:
            return -2.0
        total_active = surgeon_active_weeks[name]
        so_far       = active_so_far[name]
        budget       = t * (so_far / total_active) if total_active > 0 else 0
        pace_deficit = (budget - served[name]) / t

        # v27: behind pace, an elective surgeon has fewer weeks left to make it
        # up in, so the same shortfall matters more. Ahead of pace they yield
        # normally — scaling a surplus would push them past target, which is
        # the opposite of what the arrangement is for.
        if pace_deficit > 0:
            pace_deficit *= pace_scarcity[name]

        rest         = min((wi - last_service_wi[name]) * 0.08, 0.4)
        pref_adj     = 0.0

        if served[name] >= t:
            if pref == 'willing':
                pref_adj = 0.1
            elif pref == 'seeking':
                pref_adj = 0.2
            else:
                pref_adj = -0.8

        week_label = all_weeks[wi]['label'] if wi < len(all_weeks) else ''
        week_info  = surgeon_week_ranks.get(name, {}).get(week_label)
        if week_info:
            rank = week_info['rank']
            base = 0.6 if rank <= 2 else (0.35 if rank <= 4 else 0.15)
            mult = 1.0
            if week_info['is_holiday']:
                mult = 1.5 * holiday_claim_factor(name, holiday_history)
            pref_adj -= base * mult

        # v26: a surgeon with two weeks a month already off service cannot
        # reach target on 5-day weeks — an ACS M-F week costs a whole week for
        # five shifts. Discouraged, not barred: if it is the only way to fill
        # the role, coverage still wins.
        if role == 'ACS (M-F)' and prefers_7day.get(name):
            pref_adj -= 0.5

        return pace_deficit + rest + pref_adj

    def fallback_score(surgeon):
        name = surgeon['name']
        t    = targets[name]
        if t == 0:
            return served[name]
        return served[name] / t

    week_assignments = {}
    for wi, week in enumerate(all_weeks):
        week_assignments[wi] = {
            'label':     week['label'],
            'start':     week['start'],
            'year':      week['year'],
            'month':     week['month'],
            'month_idx': week['month_idx'],
        }

    for wi, week in enumerate(all_weeks):
        assigned_this_week = {}
        print(f"DEBUG week {wi}: {week['label']} month_idx={week['month_idx']} year={week['year']} month={week['month']}", flush=True)

        for s in surgeons:
            if is_active_for_week(s, week):
                active_so_far[s['name']] += 1

        for role in ROLE_ORDER:
            if role in assigned_this_week:
                continue
            for fellow in fellows:
                if not base_eligible(fellow, role, week, wi, assigned_this_week):
                    continue
                if fellow_can_take_role(fellow, role, week):
                    if role not in assigned_this_week:
                        assigned_this_week[role] = fellow['name']
                        break

        for role in ROLE_ORDER:
            if role in assigned_this_week:
                continue
            candidates = [
                (pace_score(s, wi, role), s['name'], s)
                for s in surgeons
                if base_eligible(s, role, week, wi, assigned_this_week)
                and within_cap(s)
            ]
            if candidates:
                candidates.sort(key=lambda x: (-x[0], x[1]))
                best = candidates[0][2]
                assigned_this_week[role] = best['name']

        for role in ROLE_ORDER:
            if role in assigned_this_week:
                continue
            fallback_candidates = [
                (fallback_score(s), s['name'], s)
                for s in surgeons
                if base_eligible(s, role, week, wi, assigned_this_week)
            ]
            if fallback_candidates:
                fallback_candidates.sort(key=lambda x: (x[0], x[1]))
                best = fallback_candidates[0][2]
                assigned_this_week[role] = best['name']

        print(f"DEBUG week {wi} assigned: {assigned_this_week}", flush=True)

        for role, name in assigned_this_week.items():
            surgeon = next((s for s in surgeons if s['name'] == name), None)
            if surgeon is None:
                continue
            served[name]          += ROLE_SHIFTS[role]
            last_service_wi[name]  = wi
            if is_seven_day_role(role):
                # Run length is read from the previous 7-day week, so it has to
                # be updated before last_7day_wi moves.
                if wi - last_7day_wi[name] == 1:
                    run_7day[name] += 1
                else:
                    run_7day[name] = 1
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

    print(f"DEBUG final served: {served}", flush=True)
    return week_assignments


def solve_call(surgeons, months, week_assignments, preferences, block_start=None):
    num_surgeons   = len(surgeons)
    num_months     = len(months)
    month_days     = [monthrange(y, mo)[1] for y, mo in months]
    fellow_indices = [i for i, s in enumerate(surgeons) if is_fellow(s)]

    # v24: days before the block's first Monday belong to the previous block's
    # last week. They are already published; the solver must not touch them.
    day_start = block_day_offsets(months, block_start)
    if day_start[0] > 0:
        print(f"DEBUG call skipping {day_start[0]} leading day(s) of "
              f"{months[0][0]}-{months[0][1]:02d} — owned by previous block", flush=True)

    surgeon_avoid = {}
    for i, s in enumerate(surgeons):
        prefs = get_surgeon_prefs(s.get('id', ''), preferences)
        surgeon_avoid[i] = parse_date_list(
            prefs.get('avoid_nights', ''), months[0][0])

    label_to_dates = {}
    for wi, wa in week_assignments.items():
        start = wa['start']
        label_to_dates[wa['label']] = [(start + timedelta(days=o)).date()
                                       for o in range(7)]

    surgeon_offweek = {i: {} for i in range(num_surgeons)}
    for i, s in enumerate(surgeons):
        prefs = get_surgeon_prefs(s.get('id', ''), preferences)
        tow   = prefs.get('time_off_weeks', [])
        if isinstance(tow, list):
            for w in tow:
                lbl = w.get('week', '')
                if lbl in label_to_dates:
                    rank = int(w.get('rank', 99))
                    hol  = bool(w.get('isHoliday', False))
                    for dt in label_to_dates[lbl]:
                        surgeon_offweek[i][dt] = (rank, hol)

    # v26: the nights an elective surgeon actually asked for. The route sends
    # their elective week as rank-1 time off, which would otherwise suppress
    # every night in it — including the three they want.
    elective_plan = build_elective_plan(surgeons, months, week_assignments, day_start)

    active_in_month = [
        [is_active_for_month(surgeons[i], y, mo) for i in range(num_surgeons)]
        for mi, (y, mo) in enumerate(months)
    ]

    night_role = {}
    for mi, (y, mo) in enumerate(months):
        night_role[mi] = {}
        for d in range(day_start[mi], month_days[mi]):
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

    # Variables are still built for every day so indexing stays simple. Days
    # outside the block are pinned to zero and never get an ExactlyOne.
    call = [
        [[model.NewBoolVar(f'c_{mi}_{d}_{i}') for i in range(num_surgeons)]
         for d in range(month_days[mi])]
        for mi in range(num_months)
    ]

    for mi in range(num_months):
        for d in range(month_days[mi]):
            if d < day_start[mi]:
                for i in range(num_surgeons):
                    model.Add(call[mi][d][i] == 0)
            else:
                model.AddExactlyOne(call[mi][d])

    for mi, (y, mo) in enumerate(months):
        for d in range(day_start[mi], month_days[mi]):
            for i in range(num_surgeons):
                if (not active_in_month[mi][i] or
                        not is_eligible(surgeons[i], 'call')):
                    model.Add(call[mi][d][i] == 0)

    for mi, (y, mo) in enumerate(months):
        for d in range(day_start[mi], month_days[mi]):
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

    for i in fellow_indices:
        max_call = int(surgeons[i].get('max_call_per_month', 5))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][i]
                    for d in range(day_start[mi], month_days[mi])) <= max_call)

    for i in range(num_surgeons):
        if i in fellow_indices:
            continue
        max_call = int(surgeons[i].get('max_call_per_month', 8))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][i]
                    for d in range(day_start[mi], month_days[mi])) <= max_call)

    obj_terms     = []
    penalty_terms = []

    weekend_nights = [
        (mi, d)
        for mi, (y, mo) in enumerate(months)
        for d in range(day_start[mi], month_days[mi])
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

    for mi, (y, mo) in enumerate(months):
        for d in range(day_start[mi], month_days[mi]):
            dow = datetime(y, mo, d + 1).weekday()
            for i in range(num_surgeons):
                pref_days = surgeons[i].get('call_day_preference', '') or ''
                if pref_days == 'friday_saturday':
                    if dow in (4, 5):
                        obj_terms.append(3 * call[mi][d][i])
                    else:
                        penalty_terms.append(2 * call[mi][d][i])

    for mi in range(num_months):
        days = month_days[mi]
        for i in range(num_surgeons):
            for d in range(day_start[mi], days - 3):
                run4 = model.NewBoolVar(f'r4_{mi}_{d}_{i}')
                model.AddMinEquality(run4, [
                    call[mi][d][i],   call[mi][d+1][i],
                    call[mi][d+2][i], call[mi][d+3][i]
                ])
                penalty_terms.append(25 * run4)

    for mi, (y, mo) in enumerate(months):
        for d in range(day_start[mi], month_days[mi]):
            for i in range(num_surgeons):
                if surgeon_avoid[i] and day_in_dates(y, mo, d, surgeon_avoid[i]):
                    penalty_terms.append(30 * call[mi][d][i])

    for mi, (y, mo) in enumerate(months):
        for d in range(day_start[mi], month_days[mi]):
            cur = datetime(y, mo, d + 1).date()
            for i in range(num_surgeons):
                # v26: a night the surgeon explicitly asked for inside their own
                # elective week is not a night off. Skipping the penalty here is
                # what stops the two instructions cancelling out.
                if (mi, d) in elective_plan.get(i, {}).get('call', ()):
                    continue
                info = surgeon_offweek.get(i, {}).get(cur)
                if not info:
                    continue
                rank, hol = info
                if rank <= 2:
                    base = CALL_OFFWEEK_PENALTY['top']
                elif rank <= 4:
                    base = CALL_OFFWEEK_PENALTY['mid']
                else:
                    base = CALL_OFFWEEK_PENALTY['low']
                if hol:
                    base = int(base * CALL_OFFWEEK_HOLIDAY_MULT)
                penalty_terms.append(base * call[mi][d][i])

    # v26: the elective shape itself. A reward on each requested night rather
    # than a hard "exactly three", so a night nobody else can cover still gets
    # covered and the week simply comes back one short — visible, not broken.
    for i, plan in elective_plan.items():
        for (mi, d) in plan['call']:
            obj_terms.append(ELECTIVE_CALL_REWARD * call[mi][d][i])
        for (mi, d) in plan['avoid']:
            penalty_terms.append(ELECTIVE_AVOID_PENALTY * call[mi][d][i])

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
            f"Check call eligibility flags in admin."
        )

    call_assignments = {}
    for mi in range(num_months):
        for d in range(day_start[mi], month_days[mi]):
            for i in range(num_surgeons):
                if solver.Value(call[mi][d][i]):
                    call_assignments[(mi, d)] = surgeons[i]['name']

    # How many of the requested elective nights were actually granted. Printed
    # rather than inferred later from a schedule nobody wants to count by hand.
    for i, plan in elective_plan.items():
        got = sum(1 for slot in plan['call']
                  if call_assignments.get(slot) == surgeons[i]['name'])
        print(f"DEBUG elective {surgeons[i]['name']}: "
              f"{got} of {len(plan['call'])} requested call nights granted",
              flush=True)

    return call_assignments


def validate_schedule(result, surgeons, months, block_number, prior_totals,
                      preferences=None, holiday_history=None, block_start=None):
    """
    THE single source of truth for flags/violations/warnings.
    Operates purely on an assembled schedule (`result`: {month_key: {weeks,
    nights}}), so it works identically for a fresh solve and for a hand-edited
    schedule sent by the app. No optimizing here — only checking.

    The missing-call check starts at the block's first Monday. Days before it
    sit inside a week the previous month owns and has already published, and
    this block never produces them — so looking for them here only ever
    generates false violations. Every other check reads whatever is present
    and needs no gating.
    """
    preferences     = preferences or []
    holiday_history = holiday_history or {}

    num_months = len(months)
    month_days = [monthrange(y, mo)[1] for y, mo in months]
    day_start  = block_day_offsets(months, block_start)

    block_targets = {
        s['name']: compute_block_target(s, block_number, prior_totals, months)
        for s in surgeons
    }
    target_shifts = {name: max(0, round(t)) for name, t in block_targets.items()}

    all_weeks = get_all_weeks(months, block_start)

    # Ordered weeks (chronological) with their assignments pulled from `result`
    # by label. Replaces the internal week_assignments dict for adjacency checks.
    label_to_week = {}
    for mk, md in result.items():
        for w in (md.get('weeks') or []):
            if w.get('label'):
                label_to_week[w['label']] = w
    ordered = [{'start': w['start'], 'end': w['end'], 'label': w['label'],
                'assign': label_to_week.get(w['label'], {})} for w in all_weeks]
    start_to_idx = {w['start']: i for i, w in enumerate(ordered)}

    # Flat chronological list of week dicts (for consecutive-week checks)
    all_weeks_flat = []
    for mi in range(num_months):
        mk = f"{months[mi][0]}-{str(months[mi][1]).zfill(2)}"
        all_weeks_flat.extend((result.get(mk, {}) or {}).get('weeks', []) or [])

    # Recompute served totals from the schedule itself (robust to hand edits)
    served_total = {s['name']: 0 for s in surgeons}
    for w in all_weeks_flat:
        for role, sc in ROLE_SHIFTS.items():
            n = w.get(role)
            if n in served_total:
                served_total[n] += sc

    violations = []
    warnings   = []
    flags      = []

    def add_flag(ftype, severity, message, **extra):
        f = {'type': ftype, 'severity': severity, 'message': message}
        f.update(extra)
        flags.append(f)
        warnings.append(message)

    # ── Structural violations: unfilled roles, missing call, double-booking ──
    for mi, (y, mo) in enumerate(months):
        mk          = f"{y}-{str(mo).zfill(2)}"
        month_label = datetime(y, mo, 1).strftime('%B %Y')
        md          = result.get(mk, {}) or {}
        weeks       = md.get('weeks', []) or []
        nights      = md.get('nights', {}) or {}

        for w in weeks:
            for role in ROLE_SHIFTS:
                if not w.get(role):
                    violations.append(
                        f"{month_label} {w.get('label','?')}: {role} unfilled — "
                        f"no eligible surgeon found")

        for d in range(day_start[mi], month_days[mi]):
            if not nights.get(str(d + 1), {}).get('Call'):
                violations.append(
                    f"{month_label} day {d + 1}: No call surgeon assigned")

        for w in weeks:
            seen = {}
            for role in ROLE_SHIFTS:
                name = w.get(role)
                if name:
                    if name in seen:
                        violations.append(
                            f"{month_label} {w.get('label','?')}: "
                            f"{name} in {seen[name]} and {role}")
                    seen[name] = role

    # ── Consecutive 7-day service weeks (warning) ──
    # v26: silent for a surgeon who has opted into two in a row. It is the
    # arrangement working, not a compromise, and flagging it would train
    # everyone to scroll past a list that is mostly noise.
    allows_pair = {s['name']: elective_max_consecutive(s) >= 2 for s in surgeons}
    seven_day_roles = ['ACS (M-Sun)', 'McNair ICU', 'TSICU', 'SICU']
    for i in range(len(all_weeks_flat) - 1):
        w1 = all_weeks_flat[i]
        w2 = all_weeks_flat[i + 1]
        for r1 in seven_day_roles:
            for r2 in seven_day_roles:
                n1 = w1.get(r1)
                n2 = w2.get(r2)
                if n1 and n2 and n1 == n2 and not allows_pair.get(n1):
                    warnings.append(
                        f"Consecutive 7-day weeks: {n1} "
                        f"({r1} -> {r2}) — review manually")

    # Three in a row is beyond any ceiling and is always worth saying.
    for i in range(len(all_weeks_flat) - 2):
        trio = [all_weeks_flat[i], all_weeks_flat[i + 1], all_weeks_flat[i + 2]]
        holders = []
        for w in trio:
            names = [w.get(r) for r in seven_day_roles if w.get(r)]
            holders.append(set(names))
        common = holders[0] & holders[1] & holders[2]
        for n in sorted(common):
            warnings.append(
                f"Three consecutive 7-day weeks: {n} — beyond any ceiling, "
                f"review manually")

    # ── ACS M-Sun consecutive (hard violation) ──
    for i in range(len(all_weeks_flat) - 1):
        n1 = all_weeks_flat[i].get('ACS (M-Sun)')
        n2 = all_weeks_flat[i + 1].get('ACS (M-Sun)')
        if n1 and n2 and n1 == n2:
            violations.append(f"ACS M-Sun consecutive: {n1} — hard rule violated")

    # ── 4+ consecutive call nights (warning) ──
    for mi, (y, mo) in enumerate(months):
        nights = (result.get(f"{y}-{str(mo).zfill(2)}", {}) or {}).get('nights', {}) or {}
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

    # ── Sunday call then fresh Monday service (warning) ──
    for mi, (y, mo) in enumerate(months):
        nights = (result.get(f"{y}-{str(mo).zfill(2)}", {}) or {}).get('nights', {}) or {}
        for d in range(month_days[mi]):
            if datetime(y, mo, d + 1).weekday() != 6:
                continue
            call_name = nights.get(str(d + 1), {}).get('Call', '')
            if not call_name:
                continue
            next_monday = datetime(y, mo, d + 1) + timedelta(days=1)
            idx = start_to_idx.get(next_monday)
            if idx is None:
                continue
            wa = ordered[idx]['assign']
            for role in ROLE_SHIFTS:
                if wa.get(role) == call_name:
                    prior = ordered[idx - 1]['assign'] if idx - 1 >= 0 else {}
                    in_prior = any(prior.get(r) == call_name for r in ROLE_SHIFTS)
                    if not in_prior:
                        warnings.append(
                            f"{call_name}: call Sun "
                            f"{next_monday.strftime('%b %-d')} "
                            f"then fresh {role} Mon — fix manually")

    # ── Fellow rotation quotas (violation) ──
    periods = get_two_month_periods(months)
    fellows = [s for s in surgeons if is_fellow(s)]
    fpt     = compute_fellow_period_targets(fellows, periods, all_weeks)

    for pi, period_months in enumerate(periods):
        for fellow in fellows:
            fname         = fellow['name']
            acs_t, sicu_t = fpt[fname].get(pi, (0, 0))
            acs_count = sicu_count = 0
            for pm in period_months:
                mk = f"{pm[0]}-{str(pm[1]).zfill(2)}"
                for w in (result.get(mk, {}) or {}).get('weeks', []) or []:
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

    # ── FTE / compensation summary (warnings + block summary) ──
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
        total = served_total.get(name, 0)
        t     = target_shifts[name]
        cap   = compute_soft_cap(t, pref)
        delta = total - t

        if t > 0 and total > cap:
            warnings.append(
                f"{name}: served {total} shifts (cap {cap}, target {t}) — "
                f"assigned beyond cap to cover unfilled roles. "
                f"Review for compensation.")
        elif t > 0 and delta < -7:
            # v26: a surgeon with an elective practice has two weeks a month off
            # service by arrangement, so falling short is the expected shape
            # rather than thin coverage. Said differently so nobody reads it as
            # a fault in the schedule.
            if get_elective(s):
                warnings.append(
                    f"{name}: served {total} vs target {t} "
                    f"(short {abs(delta)}) — elective practice reduces available "
                    f"weeks; review whether the shape reaches target.")
            else:
                warnings.append(
                    f"{name}: served {total} vs target {t} "
                    f"(short {abs(delta)}) — insufficient eligible coverage")

        block_fte_summary[name] = {
            'served': total,
            'target': round(block_targets[name], 1),
            'delta':  round(delta, 1),
        }

        count = sum(
            1 for mi, d in weekend_nights
            if (result.get(f"{months[mi][0]}-{str(months[mi][1]).zfill(2)}", {}) or {})
               .get('nights', {}).get(str(d + 1), {}).get('Call') == name
        )
        if count > 0:
            weekend_call_summary[name] = count

    # ── Preference-honoring flags (typed) ──
    surgeon_week_ranks = build_week_ranks(surgeons, preferences)

    def week_label_for_date(dt):
        for w in all_weeks:
            if w['start'] <= dt <= w['end']:
                return w['label']
        return None

    # 1) Service overrides
    for mi, (y, mo) in enumerate(months):
        mk = f"{y}-{str(mo).zfill(2)}"
        for w in (result.get(mk, {}) or {}).get('weeks', []) or []:
            lbl = w.get('label')
            for role in ROLE_SHIFTS:
                name = w.get(role)
                if not name or not lbl:
                    continue
                info = surgeon_week_ranks.get(name, {}).get(lbl)
                if info:
                    is_hol = info['is_holiday']
                    rank   = info['rank']
                    src    = info.get('source', '')
                    if src in ('elective', 'admin'):
                        # An elective or admin week filled anyway is a real
                        # compromise, but it is not the surgeon's ranked
                        # holiday request and should not read as one.
                        add_flag(
                            'elective_override', 'high',
                            f"ELECTIVE OVERRIDE: {name} assigned {role} for {lbl} "
                            f"({'elective' if src == 'elective' else 'admin'} week) "
                            f"— coverage forced; exec review.",
                            surgeon=name, week=lbl, role=role, rank=rank,
                            is_holiday=False, elective_kind=src)
                    else:
                        ftype = 'holiday_override' if is_hol else 'preference_override'
                        tag   = 'HOLIDAY OVERRIDE' if is_hol else 'PREF OVERRIDE'
                        add_flag(
                            ftype, 'high',
                            f"{tag}: {name} assigned {role} for {lbl} "
                            f"(requested off, rank #{rank}) — coverage forced; "
                            f"exec review.",
                            surgeon=name, week=lbl, role=role, rank=rank,
                            is_holiday=is_hol)

    # 2) Contested weeks
    # An elective or admin week is a standing arrangement, not a bid for a week,
    # so it does not make a week contested — otherwise every elective surgeon
    # would appear to be fighting the whole division six times a block.
    week_requesters = {}
    for name, ranks in surgeon_week_ranks.items():
        for lbl, info in ranks.items():
            if info.get('source') in ('elective', 'admin'):
                continue
            week_requesters.setdefault(lbl, []).append(
                (name, info['rank'], info['is_holiday']))
    for lbl, reqs in week_requesters.items():
        if len(reqs) >= 2:
            is_hol      = any(r[2] for r in reqs)
            contenders  = sorted(reqs, key=lambda r: r[1])
            names_ranks = ", ".join(f"{n} (#{rk})" for n, rk, _ in contenders)
            adjudication = None
            if is_hol and holiday_history:
                adjudication = [
                    n for n, _, _ in sorted(
                        contenders,
                        key=lambda r: (holiday_history.get(r[0], {}) or {})
                        .get('recent_holidays_worked', 0),
                        reverse=True)
                ]
            add_flag(
                'contested_week', 'medium',
                f"CONTESTED WEEK{' [HOLIDAY]' if is_hol else ''}: {lbl} "
                f"requested by {len(reqs)} surgeons — {names_ranks}. "
                f"Filled; chief to review.",
                week=lbl, is_holiday=is_hol,
                contenders=[{'surgeon': n, 'rank': rk, 'is_holiday': h}
                            for n, rk, h in contenders],
                adjudication=adjudication)

    # 3) Call during a requested week off
    # Call inside an elective week is the arrangement, not a breach — the whole
    # point is a small number of nights inside a week off service. Call inside
    # an ADMIN week is still wrong and still flagged.
    for mi, (y, mo) in enumerate(months):
        mk     = f"{y}-{str(mo).zfill(2)}"
        nights = (result.get(mk, {}) or {}).get('nights', {}) or {}
        for d in range(month_days[mi]):
            call_name = nights.get(str(d + 1), {}).get('Call', '')
            if not call_name:
                continue
            dt  = datetime(y, mo, d + 1)
            lbl = week_label_for_date(dt)
            if not lbl:
                continue
            info = surgeon_week_ranks.get(call_name, {}).get(lbl)
            if not info:
                continue
            src = info.get('source', '')
            if src == 'elective':
                continue
            if src == 'admin':
                add_flag(
                    'call_during_admin_week', 'high',
                    f"CALL IN ADMIN WEEK: {call_name} on call "
                    f"{dt.strftime('%b %-d')} during an admin week ({lbl}) — "
                    f"that week is meant to hold no call.",
                    surgeon=call_name, date=dt.strftime('%Y-%m-%d'),
                    week=lbl, rank=info['rank'], is_holiday=False)
                continue
            add_flag(
                'call_during_week_off', 'high',
                f"CALL DURING WEEK OFF: {call_name} on call "
                f"{dt.strftime('%b %-d')} during requested week off {lbl} "
                f"(rank #{info['rank']}) — exec decision required.",
                surgeon=call_name, date=dt.strftime('%Y-%m-%d'),
                week=lbl, rank=info['rank'], is_holiday=info['is_holiday'])

    return {
        'violations':           violations,
        'warnings':             warnings,
        'flags':                flags,
        'valid':                len(violations) == 0,
        'block_fte_summary':    block_fte_summary,
        'weekend_call_summary': weekend_call_summary,
    }


def build_output(surgeons, months, week_assignments, call_assignments,
                 block_number, prior_totals, preferences=None,
                 holiday_history=None, block_start=None):

    preferences     = preferences or []
    holiday_history = holiday_history or {}

    num_months = len(months)
    month_days = [monthrange(y, mo)[1] for y, mo in months]

    # v24: leading days owned by the previous block are omitted entirely, so a
    # merge upstream cannot overwrite published nights with empty values.
    day_start = block_day_offsets(months, block_start)

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
            for role in ROLE_SHIFTS:
                week_data[role] = wa.get(role, '')
            result_weeks.append(week_data)

        result_nights = {}
        for d in range(day_start[mi], month_days[mi]):
            name = call_assignments.get((mi, d), '')
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

    # Single source of truth — same function the /validate-only endpoint uses.
    validation = validate_schedule(
        result, surgeons, months, block_number, prior_totals,
        preferences=preferences, holiday_history=holiday_history,
        block_start=block_start)

    return {'months': result, 'validation': validation}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
