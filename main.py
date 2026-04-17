"""
ORbit Surgical Scheduling Solver v21
=====================================
v21: Surgeon ranked time-off week preferences now applied
as soft penalties in pace_score. Rank 1-2 = strong penalty,
Rank 3-4 = moderate, Rank 5+ = light. Holiday weeks get
1.5x penalty multiplier. Preferences are soft only —
all roles must be filled regardless.

v20: Block target = 84 × FTE for every block, always.
No carry-forward from Block 1. Every shift over target
is compensation, tracked independently per block.
Prorated only for start/departure dates.
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


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ORbit Solver v21'})


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

        print("=== v21 SOLVER STARTED ===", flush=True)
        print(f"DEBUG block_number={block_number} start_year={start_year} months={months}", flush=True)
        for s in surgeons:
            print(f"  {s['name']} | is_fellow={s.get('is_fellow')} | fte={s.get('fte')} | sicu={s.get('covers_sicu')} | acs={s.get('can_acs')}", flush=True)

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


def get_all_weeks(months):
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


def greedy_service_weeks(surgeons, months, block_number, preferences, prior_totals):
    all_weeks = get_all_weeks(months)
    periods   = get_two_month_periods(months)

    fellows   = [s for s in surgeons if is_fellow(s)]
    all_names = [s['name'] for s in surgeons]

    print(f"DEBUG fellows detected: {[f['name'] for f in fellows]}", flush=True)
    print(f"DEBUG total weeks generated: {len(all_weeks)}", flush=True)

    # ── Build time-off sets (legacy free-text, kept for compatibility) ────────
    surgeon_time_off = {}
    for s in surgeons:
        prefs = get_surgeon_prefs(s.get('id', ''), preferences)
        y_ref = months[0][0]
        off   = parse_date_list(prefs.get('time_off',    ''), y_ref)
        conf  = parse_date_list(prefs.get('conferences', ''), y_ref)
        surgeon_time_off[s['name']] = off | conf

    # ── Build ranked week preference dict (new structured format) ─────────────
    # surgeon_week_ranks[surgeon_name][week_label] = {rank, is_holiday}
    surgeon_week_ranks = {}
    for s in surgeons:
        prefs          = get_surgeon_prefs(s.get('id', ''), preferences)
        time_off_weeks = prefs.get('time_off_weeks', [])
        if isinstance(time_off_weeks, list) and time_off_weeks:
            surgeon_week_ranks[s['name']] = {
                w.get('week', ''): {
                    'rank':       int(w.get('rank', 99)),
                    'is_holiday': bool(w.get('isHoliday', False)),
                }
                for w in time_off_weeks
                if w.get('week')
            }
            print(
                f"DEBUG {s['name']} time_off_weeks: "
                f"{list(surgeon_week_ranks[s['name']].keys())}",
                flush=True
            )
        else:
            surgeon_week_ranks[s['name']] = {}

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

    served           = {n: 0   for n in all_names}
    last_service_wi  = {n: -99 for n in all_names}
    last_7day_wi     = {n: -99 for n in all_names}
    last_acs_msun_wi = {n: -99 for n in all_names}
    active_so_far    = {n: 0   for n in all_names}

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
        if is_seven_day_role(role) and wi - last_7day_wi[name] <= 1:
            return False
        if role == 'ACS (M-Sun)' and wi - last_acs_msun_wi[name] <= 1:
            return False
        if is_fellow(surgeon) and not fellow_can_take_role(surgeon, role, week):
            return False
        return True

    def within_cap(surgeon):
        name = surgeon['name']
        return served[name] < soft_caps[name]

    def pace_score(surgeon, wi):
        name  = surgeon['name']
        t     = targets[name]
        pref  = get_pref(surgeon)
        if t == 0:
            return -2.0
        total_active = surgeon_active_weeks[name]
        so_far       = active_so_far[name]
        budget       = t * (so_far / total_active) if total_active > 0 else 0
        pace_deficit = (budget - served[name]) / t
        rest         = min((wi - last_service_wi[name]) * 0.08, 0.4)
        pref_adj     = 0.0

        # Over-target penalty by preference tier
        if served[name] >= t:
            if pref == 'willing':
                pref_adj = 0.1
            elif pref == 'seeking':
                pref_adj = 0.2
            else:
                pref_adj = -0.8

        # ── Ranked time-off week penalty (v21) ────────────────────────────────
        # Reduces score for weeks surgeon has requested off.
        # Penalty scales with rank (1=strongest) and holiday multiplier.
        # This is a soft constraint — role must still be filled if no one else
        # is available (fallback pass ignores this and assigns anyway).
        week_label = all_weeks[wi]['label'] if wi < len(all_weeks) else ''
        week_info  = surgeon_week_ranks.get(name, {}).get(week_label)
        if week_info:
            rank         = week_info['rank']
            holiday_mult = 1.5 if week_info['is_holiday'] else 1.0
            if rank <= 2:
                pref_adj -= 0.6 * holiday_mult
            elif rank <= 4:
                pref_adj -= 0.35 * holiday_mult
            else:
                pref_adj -= 0.15 * holiday_mult

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

        # Pass 1 — Fellows get priority for their rotation requirements
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

        # Pass 2 — Within-cap greedy with pace + preference scoring
        for role in ROLE_ORDER:
            if role in assigned_this_week:
                continue
            candidates = [
                (pace_score(s, wi), s['name'], s)
                for s in surgeons
                if base_eligible(s, role, week, wi, assigned_this_week)
                and within_cap(s)
            ]
            if candidates:
                candidates.sort(key=lambda x: (-x[0], x[1]))
                best = candidates[0][2]
                assigned_this_week[role] = best['name']

        # Pass 3 — Fallback: assign least-loaded eligible regardless of cap
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


def solve_call(surgeons, months, week_assignments, preferences):
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

    for mi in range(num_months):
        for d in range(month_days[mi]):
            model.AddExactlyOne(call[mi][d])

    for mi, (y, mo) in enumerate(months):
        for d in range(month_days[mi]):
            for i in range(num_surgeons):
                if (not active_in_month[mi][i] or
                        not is_eligible(surgeons[i], 'call')):
                    model.Add(call[mi][d][i] == 0)

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

    for i in fellow_indices:
        max_call = int(surgeons[i].get('max_call_per_month', 5))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][i] for d in range(month_days[mi])) <= max_call)

    for i in range(num_surgeons):
        if i in fellow_indices:
            continue
        max_call = int(surgeons[i].get('max_call_per_month', 8))
        for mi in range(num_months):
            model.Add(
                sum(call[mi][d][i] for d in range(month_days[mi])) <= max_call)

    obj_terms     = []
    penalty_terms = []

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
            f"Check call eligibility flags in admin."
        )

    call_assignments = {}
    for mi in range(num_months):
        for d in range(month_days[mi]):
            for i in range(num_surgeons):
                if solver.Value(call[mi][d][i]):
                    call_assignments[(mi, d)] = surgeons[i]['name']

    return call_assignments


def build_output(surgeons, months, week_assignments, call_assignments,
                 block_number, prior_totals):

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
        for d in range(month_days[mi]):
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
                    violations.append(
                        f"{month_label} {w['label']}: {role} unfilled — "
                        f"no eligible surgeon found")

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

    for i in range(len(all_weeks_flat) - 1):
        n1 = all_weeks_flat[i].get('ACS (M-Sun)')
        n2 = all_weeks_flat[i + 1].get('ACS (M-Sun)')
        if n1 and n2 and n1 == n2:
            violations.append(f"ACS M-Sun consecutive: {n1} — hard rule violated")

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
        cap   = compute_soft_cap(t, pref)
        delta = total - t

        if t > 0 and total > cap:
            warnings.append(
                f"{name}: served {total} shifts (cap {cap}, target {t}) — "
                f"assigned beyond cap to cover unfilled roles. "
                f"Review for compensation.")
        elif t > 0 and delta < -7:
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
