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

def get_weeks(year, month):
    first_day = datetime(year, month, 1)
    dow = first_day.weekday()
    week_start = first_day - timedelta(days=dow)
    weeks = []
    while True:
        week_end = week_start + timedelta(days=6)
        if week_start.month > month and week_start.year >= year:
            break
        if week_end.month < month and week_end.year <= year:
            week_start += timedelta(days=7)
            continue
        weeks.append({
            'start': week_start,
            'end': week_end,
            'label': f"{week_start.strftime('%b %-d')} - {week_end.strftime('%b %-d')}"
        })
        week_start += timedelta(days=7)
        if week_start.month > month and week_start.year >= year:
            break
    return weeks

def solve_month(surgeons, year, month, preferences, prior_totals, block_number):
    days_in_month = monthrange(year, month)[1]
    weeks = get_weeks(year, month)
    num_weeks = len(weeks)
    num_days = days_in_month
    num_surgeons = len(surgeons)

    def eligible(s, role):
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

    # Filter inactive surgeons (not yet started)
    inactive = set()
    for s in range(num_surgeons):
        start = surgeons[s].get('start_date', '')
        if start:
            try:
                start_dt = datetime.strptime(start[:10], '%Y-%m-%d')
                if start_dt.year > year or (start_dt.year == year and start_dt.month > month):
                    inactive.add(s)
            except:
                pass

    fellow_indices = [
        s for s in range(num_surgeons)
        if 'fellow' in surgeons[s].get('name', '').lower()
        and s not in inactive
    ]

    model = cp_model.CpModel()

    acs_msun = [[model.NewBoolVar(f'am_{w}_{s}') for s in range(num_surgeons)] for w in range(num_weeks)]
    acs_mf   = [[model.NewBoolVar(f'af_{w}_{s}') for s in range(num_surgeons)] for w in range(num_weeks)]
    mcnair   = [[model.NewBoolVar(f'mn_{w}_{s}') for s in range(num_surgeons)] for w in range(num_weeks)]
    tsicu    = [[model.NewBoolVar(f'ts_{w}_{s}') for s in range(num_surgeons)] for w in range(num_weeks)]
    sicu     = [[model.NewBoolVar(f'si_{w}_{s}') for s in range(num_surgeons)] for w in range(num_weeks)]
    call     = [[model.NewBoolVar(f'ca_{d}_{s}') for s in range(num_surgeons)] for d in range(num_days)]

    # Exactly 1 per weekly role
    for w in range(num_weeks):
        model.AddExactlyOne(acs_msun[w])
        model.AddExactlyOne(acs_mf[w])
        model.AddExactlyOne(mcnair[w])
        model.AddExactlyOne(tsicu[w])
        model.AddExactlyOne(sicu[w])

    # Exactly 1 call per night
    for d in range(num_days):
        model.AddExactlyOne(call[d])

    # Eligibility and inactive
    for w in range(num_weeks):
        for s in range(num_surgeons):
            if s in inactive or not eligible(surgeons[s], 'acs_msun'):
                model.Add(acs_msun[w][s] == 0)
            if s in inactive or not eligible(surgeons[s], 'acs_mf'):
                model.Add(acs_mf[w][s] == 0)
            if s in inactive or not eligible(surgeons[s], 'mcnair'):
                model.Add(mcnair[w][s] == 0)
            if s in inactive or not eligible(surgeons[s], 'tsicu'):
                model.Add(tsicu[w][s] == 0)
            if s in inactive or not eligible(surgeons[s], 'sicu'):
                model.Add(sicu[w][s] == 0)

    for d in range(num_days):
        for s in range(num_surgeons):
            if s in inactive or not eligible(surgeons[s], 'call'):
                model.Add(call[d][s] == 0)

    # One role per surgeon per week
    for w in range(num_weeks):
        for s in range(num_surgeons):
            model.Add(acs_msun[w][s] + acs_mf[w][s] + mcnair[w][s] + tsicu[w][s] + sicu[w][s] <= 1)

    # No consecutive 7-day weeks
    seven_day = [acs_msun, mcnair, tsicu, sicu]
    for w in range(num_weeks - 1):
        for s in range(num_surgeons):
            for r1 in seven_day:
                for r2 in seven_day:
                    model.Add(r1[w][s] + r2[w+1][s] <= 1)

    # ACS M-Sun no repeat next week
    for w in range(num_weeks - 1):
        for s in range(num_surgeons):
            model.Add(acs_msun[w][s] + acs_msun[w+1][s] <= 1)

    # Call restrictions
    for w in range(num_weeks):
        week_start = weeks[w]['start']
        for d_offset in range(7):
            actual_date = week_start + timedelta(days=d_offset)
            if actual_date.month != month or actual_date.year != year:
                continue
            d = actual_date.day - 1
            if d >= num_days:
                continue
            dow = actual_date.weekday()
            for s in range(num_surgeons):
                model.Add(mcnair[w][s] + call[d][s] <= 1)
                if dow <= 5:
                    model.Add(tsicu[w][s] + call[d][s] <= 1)
                    model.Add(sicu[w][s] + call[d][s] <= 1)
                    model.Add(acs_msun[w][s] + call[d][s] <= 1)
                if dow <= 3:
                    model.Add(acs_mf[w][s] + call[d][s] <= 1)

    # Max call nights
    for s in range(num_surgeons):
        model.Add(sum(call[d][s] for d in range(num_days)) <= surgeons[s].get('max_call_per_month', 8))

    # Max 1 weekend call night
    weekend_days = [d for d in range(num_days) if datetime(year, month, d+1).weekday() >= 5]
    for s in range(num_surgeons):
        if weekend_days:
            model.Add(sum(call[d][s] for d in weekend_days) <= 1)

    # Fellows not on same role same week
    if len(fellow_indices) >= 2:
        for w in range(num_weeks):
            for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                model.Add(sum(role[w][f] for f in fellow_indices) <= 1)

    # Max 2 ACS M-F weeks per surgeon per month
    for s in range(num_surgeons):
        model.Add(sum(acs_mf[w][s] for w in range(num_weeks)) <= 2)

    # Max 2 ACS M-Sun weeks per surgeon per month
    for s in range(num_surgeons):
        model.Add(sum(acs_msun[w][s] for w in range(num_weeks)) <= 2)

    # Simple objective: spread assignments evenly
    # Count total assignments per surgeon and minimize variance
    total_assignments = []
    for s in range(num_surgeons):
        if s in inactive:
            continue
        surgeon_weeks = []
        for w in range(num_weeks):
            surgeon_weeks.extend([acs_msun[w][s], acs_mf[w][s], mcnair[w][s], tsicu[w][s], sicu[w][s]])
        total_assignments.append(sum(surgeon_weeks))

    # Maximize total assignments weighted by FTE
    obj_terms = []
    for s in range(num_surgeons):
        if s in inactive:
            continue
        fte = surgeons[s].get('fte', 1.0)
        weight = max(1, int(fte * 10))
        for w in range(num_weeks):
            for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                if role[w][s].Name().startswith(('am_', 'af_', 'mn_', 'ts_', 'si_')):
                    obj_terms.append(weight * role[w][s])

    if obj_terms:
        model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.num_search_workers = 2

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise Exception(
            f"No valid schedule found for {year}-{month:02d}. "
            f"Status: {solver.StatusName(status)}."
        )

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
                result_nights[str(d+1)] = {'Call': surgeons[s]['name'], 'Backup': ''}

    violations = []
    warnings = []

    for w, week in enumerate(result_weeks):
        for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
            if role not in week:
                violations.append(f"Week {w+1}: {role} not assigned")

    for d in range(num_days):
        if str(d+1) not in result_nights:
            violations.append(f"Day {d+1}: No call surgeon assigned")

    fte_summary = {}
    for s in range(num_surgeons):
        name = surgeons[s]['name']
        shifts = 0
        for w in result_weeks:
            if w.get('ACS (M-F)') == name: shifts += 5
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
