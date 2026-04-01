from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from calendar import monthrange
import os

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ORbit Solver'})

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        surgeons = data.get('surgeons', [])
        year = data.get('year')
        month = data.get('month')
        preferences = data.get('preferences', [])
        prior_totals = data.get('prior_totals', {})
        block_number = data.get('block_number', 1)
        result = solve_month(surgeons, year, month, preferences, prior_totals, block_number)
        return jsonify({'success': True, 'schedule': result})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


def get_weeks_for_month(year, month):
    """
    Returns all Mon-Sun weeks that overlap with the given month.
    Each week dict has: start (datetime), end (datetime), label (str),
    and days_in_month (list of 0-indexed day numbers that fall in this month).
    """
    days_in_month = monthrange(year, month)[1]
    first_day = datetime(year, month, 1)
    
    # Find the Monday on or before the 1st
    dow = first_day.weekday()  # 0=Mon
    week_start = first_day - timedelta(days=dow)
    
    weeks = []
    while True:
        week_end = week_start + timedelta(days=6)
        
        # Find which days of this month fall in this week
        days_in_week = []
        for offset in range(7):
            d = week_start + timedelta(days=offset)
            if d.year == year and d.month == month:
                days_in_week.append(d.day - 1)  # 0-indexed
        
        if days_in_week:
            weeks.append({
                'start': week_start,
                'end': week_end,
                'label': f"{week_start.strftime('%b %-d')} - {week_end.strftime('%b %-d')}",
                'days_in_month': days_in_week,
            })
        
        week_start += timedelta(days=7)
        
        # Stop when we've passed the end of the month
        if week_start.year > year or (week_start.year == year and week_start.month > month):
            break
    
    return weeks


def solve_month(surgeons, year, month, preferences, prior_totals, block_number):
    days_in_month = monthrange(year, month)[1]
    weeks = get_weeks_for_month(year, month)
    num_weeks = len(weeks)
    num_days = days_in_month
    num_surgeons = len(surgeons)

    # ── ELIGIBILITY ──────────────────────────────────────────────
    def elig(s, role):
        if role in ('acs_msun', 'acs_mf'):
            return bool(s.get('can_acs', False))
        if role == 'mcnair':
            return bool(s.get('covers_mcnair', False))
        if role == 'tsicu':
            return bool(s.get('covers_tsicu', False))
        if role == 'sicu':
            return bool(s.get('covers_sicu', False))
        if role == 'call':
            return bool(s.get('can_call', False))
        return False

    # ── INACTIVE SURGEONS (not yet started) ──────────────────────
    inactive = set()
    for s in range(num_surgeons):
        start_str = surgeons[s].get('start_date', '')
        if start_str:
            try:
                sd = datetime.strptime(start_str[:10], '%Y-%m-%d')
                # Surgeon hasn't started yet if their start month is after this month
                if sd.year > year or (sd.year == year and sd.month > month):
                    inactive.add(s)
            except:
                pass

    # ── FELLOWS ──────────────────────────────────────────────────
    fellow_indices = [
        s for s in range(num_surgeons)
        if 'fellow' in surgeons[s].get('name', '').lower()
        and s not in inactive
    ]

    # ── FTE TARGET PER MONTH ─────────────────────────────────────
    # Block 1 (Jul-Dec): starts at zero, target = (168 * fte) / 2 / 6 per month
    # Block 2 (Jan-Jun): reads prior totals, target = remaining / 6 per month
    def monthly_target(s_idx):
        s = surgeons[s_idx]
        fte = float(s.get('fte', 1.0))
        annual = 168.0 * fte
        if block_number == 1:
            block_target = annual / 2.0
        else:
            prior = float(prior_totals.get(s.get('name', ''), 0))
            block_target = max(0.0, annual - prior)
        return block_target / 6.0

    # ── MODEL ────────────────────────────────────────────────────
    model = cp_model.CpModel()

    # Weekly role variables
    acs_msun = [[model.NewBoolVar(f'am_{w}_{s}') for s in range(num_surgeons)] for w in range(num_weeks)]
    acs_mf   = [[model.NewBoolVar(f'af_{w}_{s}') for s in range(num_surgeons)] for w in range(num_weeks)]
    mcnair   = [[model.NewBoolVar(f'mn_{w}_{s}') for s in range(num_surgeons)] for w in range(num_weeks)]
    tsicu    = [[model.NewBoolVar(f'ts_{w}_{s}') for s in range(num_surgeons)] for w in range(num_weeks)]
    sicu     = [[model.NewBoolVar(f'si_{w}_{s}') for s in range(num_surgeons)] for w in range(num_weeks)]

    # Nightly call variables
    call = [[model.NewBoolVar(f'ca_{d}_{s}') for s in range(num_surgeons)] for d in range(num_days)]

    # ── HARD CONSTRAINTS ─────────────────────────────────────────

    # 1. Exactly one surgeon per weekly role
    for w in range(num_weeks):
        model.AddExactlyOne(acs_msun[w])
        model.AddExactlyOne(acs_mf[w])
        model.AddExactlyOne(mcnair[w])
        model.AddExactlyOne(tsicu[w])
        model.AddExactlyOne(sicu[w])

    # 2. Exactly one call surgeon per night
    for d in range(num_days):
        model.AddExactlyOne(call[d])

    # 3. Zero out ineligible and inactive surgeons
    for w in range(num_weeks):
        for s in range(num_surgeons):
            if s in inactive or not elig(surgeons[s], 'acs_msun'):
                model.Add(acs_msun[w][s] == 0)
            if s in inactive or not elig(surgeons[s], 'acs_mf'):
                model.Add(acs_mf[w][s] == 0)
            if s in inactive or not elig(surgeons[s], 'mcnair'):
                model.Add(mcnair[w][s] == 0)
            if s in inactive or not elig(surgeons[s], 'tsicu'):
                model.Add(tsicu[w][s] == 0)
            if s in inactive or not elig(surgeons[s], 'sicu'):
                model.Add(sicu[w][s] == 0)

    for d in range(num_days):
        for s in range(num_surgeons):
            if s in inactive or not elig(surgeons[s], 'call'):
                model.Add(call[d][s] == 0)

    # 4. No surgeon holds more than one weekly role per week
    for w in range(num_weeks):
        for s in range(num_surgeons):
            model.Add(
                acs_msun[w][s] + acs_mf[w][s] + mcnair[w][s] +
                tsicu[w][s] + sicu[w][s] <= 1
            )

    # 5. No two consecutive 7-day service weeks (any combination)
    seven_day_roles = [acs_msun, mcnair, tsicu, sicu]
    for w in range(num_weeks - 1):
        for s in range(num_surgeons):
            for r1 in seven_day_roles:
                for r2 in seven_day_roles:
                    model.Add(r1[w][s] + r2[w + 1][s] <= 1)

    # 6. ACS M-Sun cannot repeat the following week
    for w in range(num_weeks - 1):
        for s in range(num_surgeons):
            model.Add(acs_msun[w][s] + acs_msun[w + 1][s] <= 1)

    # 7. Max 2 ACS M-F weeks per surgeon per month
    for s in range(num_surgeons):
        model.Add(sum(acs_mf[w][s] for w in range(num_weeks)) <= 2)

    # 8. Max 2 ACS M-Sun weeks per surgeon per month
    for s in range(num_surgeons):
        model.Add(sum(acs_msun[w][s] for w in range(num_weeks)) <= 2)

    # 9. Fellows cannot share same role in same week
    if len(fellow_indices) >= 2:
        for w in range(num_weeks):
            for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                model.Add(sum(role[w][f] for f in fellow_indices) <= 1)

    # 10. Call restrictions based on weekly role
    # For each week, identify which days of this month are in that week
    # and apply call restrictions accordingly
    for w in range(num_weeks):
        week_start = weeks[w]['start']
        for offset in range(7):
            day_dt = week_start + timedelta(days=offset)
            # Only apply if this day is in the current month
            if day_dt.year != year or day_dt.month != month:
                continue
            d = day_dt.day - 1  # 0-indexed
            dow = day_dt.weekday()  # 0=Mon, 6=Sun

            for s in range(num_surgeons):
                # McNair: NO call any night that week (24/7 commitment)
                model.Add(mcnair[w][s] + call[d][s] <= 1)

                # TSICU: no call Mon-Sat (Sun last resort — handled by objective)
                if dow <= 5:
                    model.Add(tsicu[w][s] + call[d][s] <= 1)

                # SICU: no call Mon-Sat (Sun last resort)
                if dow <= 5:
                    model.Add(sicu[w][s] + call[d][s] <= 1)

                # ACS M-Sun: no call Mon-Sat
                if dow <= 5:
                    model.Add(acs_msun[w][s] + call[d][s] <= 1)

                # ACS M-F: no call Mon-Thu
                if dow <= 3:
                    model.Add(acs_mf[w][s] + call[d][s] <= 1)

    # 11. Max call nights per month per surgeon
    for s in range(num_surgeons):
        max_call = int(surgeons[s].get('max_call_per_month', 8))
        model.Add(sum(call[d][s] for d in range(num_days)) <= max_call)

    # 12. Max 1 weekend call night per month per surgeon
    weekend_days = [
        d for d in range(num_days)
        if datetime(year, month, d + 1).weekday() >= 5
    ]
    for s in range(num_surgeons):
        if weekend_days:
            model.Add(sum(call[d][s] for d in weekend_days) <= 1)

    # ── SOFT OBJECTIVE ───────────────────────────────────────────
    # Maximize FTE-weighted assignments
    # Penalize preferences violations
    obj = []
    penalties = []

    for s in range(num_surgeons):
        if s in inactive:
            continue
        target = monthly_target(s)
        weight = max(1, int(target * 10))
        pref = surgeons[s].get('extra_shift_preference', 'baseline')

        for w in range(num_weeks):
            if elig(surgeons[s], 'acs_msun'):
                obj.append(weight * acs_msun[w][s])
            if elig(surgeons[s], 'acs_mf'):
                obj.append(weight * acs_mf[w][s])
            if elig(surgeons[s], 'mcnair'):
                obj.append(weight * mcnair[w][s])
            if elig(surgeons[s], 'tsicu'):
                obj.append(weight * tsicu[w][s])
            if elig(surgeons[s], 'sicu'):
                obj.append(weight * sicu[w][s])

        # Penalize over-assignment for baseline surgeons
        if pref == 'baseline':
            for w in range(num_weeks):
                for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                    penalties.append(role[w][s])

    # Penalize Rojas-Khalil on non-Fri/Sat call
    for s in range(num_surgeons):
        if 'Rojas' in surgeons[s].get('name', '') and s not in inactive:
            for d in range(num_days):
                if datetime(year, month, d + 1).weekday() not in (4, 5):
                    penalties.append(call[d][s])

    # Penalize Al-Aref SICU (prefers TSICU)
    for s in range(num_surgeons):
        if 'Al-Aref' in surgeons[s].get('name', '') and s not in inactive:
            for w in range(num_weeks):
                penalties.append(sicu[w][s])

    if obj:
        model.Maximize(sum(obj) - 5 * sum(penalties))

    # ── SOLVE ────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.num_search_workers = 2

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise Exception(
            f"No valid schedule found for {year}-{month:02d}. "
            f"Status: {solver.StatusName(status)}. "
            f"Check that enough surgeons are eligible for each role."
        )

    # ── BUILD OUTPUT ─────────────────────────────────────────────
    result_weeks = []
    for w in range(num_weeks):
        week_data = {'label': weeks[w]['label']}
        for s in range(num_surgeons):
            name = surgeons[s]['name']
            if solver.Value(acs_msun[w][s]): week_data['ACS (M-Sun)'] = name
            if solver.Value(acs_mf[w][s]):   week_data['ACS (M-F)']   = name
            if solver.Value(mcnair[w][s]):   week_data['McNair ICU']  = name
            if solver.Value(tsicu[w][s]):    week_data['TSICU']        = name
            if solver.Value(sicu[w][s]):     week_data['SICU']         = name
        result_weeks.append(week_data)

    result_nights = {}
    for d in range(num_days):
        for s in range(num_surgeons):
            if solver.Value(call[d][s]):
                result_nights[str(d + 1)] = {
                    'Call': surgeons[s]['name'],
                    'Backup': ''
                }

    # ── VALIDATION REPORT ────────────────────────────────────────
    violations = []
    warnings = []

    # Check all weekly roles filled
    for w, week in enumerate(result_weeks):
        for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
            if role not in week:
                violations.append(f"Week {w + 1} ({week['label']}): {role} not assigned")

    # Check all nights covered
    for d in range(num_days):
        if str(d + 1) not in result_nights:
            violations.append(f"Day {d + 1}: No call surgeon assigned")

    # Check no surgeon in two roles same week
    for w, week in enumerate(result_weeks):
        seen = {}
        for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
            name = week.get(role)
            if name:
                if name in seen:
                    violations.append(f"Week {w+1}: {name} assigned to both {seen[name]} and {role}")
                seen[name] = role

    # Check fellows not doubled
    for w, week in enumerate(result_weeks):
        fellow_names = []
        for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
            name = week.get(role, '')
            if 'fellow' in name.lower():
                fellow_names.append(name)
        if len(fellow_names) != len(set(fellow_names)):
            violations.append(f"Week {w+1}: Same fellow in multiple roles")

    # Check inactive surgeons not assigned
    for w, week in enumerate(result_weeks):
        for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
            name = week.get(role, '')
            for s in inactive:
                if surgeons[s]['name'] == name:
                    violations.append(f"Week {w+1}: {name} assigned but not yet active")

    # FTE summary for this month
    fte_summary = {}
    for s in range(num_surgeons):
        name = surgeons[s]['name']
        shifts = 0
        for w in result_weeks:
            if w.get('ACS (M-F)') == name:   shifts += 5
            if w.get('ACS (M-Sun)') == name: shifts += 7
            for role in ['McNair ICU', 'TSICU', 'SICU']:
                if w.get(role) == name: shifts += 7
        fte_summary[name] = shifts

    return {
        'weeks': result_weeks,
        'nights': result_nights,
        'validation': {
            'violations': violations,
            'warnings': warnings,
            'valid': len(violations) == 0,
            'fte_summary': fte_summary
        }
    }


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
