from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model
import json
from datetime import datetime, timedelta

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
        
        result = solve_month(surgeons, year, month, preferences, prior_totals)
        return jsonify({'success': True, 'schedule': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def get_weeks(year, month):
    from calendar import monthrange
    days_in_month = monthrange(year, month)[1]
    first_day = datetime(year, month, 1)
    
    # Find first Monday on or before the 1st
    dow = first_day.weekday()
    week_start = first_day - timedelta(days=dow)
    
    weeks = []
    while week_start.month <= month and week_start.year <= year:
        week_end = week_start + timedelta(days=6)
        weeks.append({
            'start': week_start,
            'end': week_end,
            'label': f"{week_start.strftime('%b %-d')} - {week_end.strftime('%b %-d')}"
        })
        week_start += timedelta(days=7)
        if week_start.month > month and week_start.year >= year:
            break
    return weeks

def solve_month(surgeons, year, month, preferences, prior_totals):
    from calendar import monthrange
    days_in_month = monthrange(year, month)[1]
    weeks = get_weeks(year, month)
    num_weeks = len(weeks)
    num_days = days_in_month
    num_surgeons = len(surgeons)

    # Build eligibility maps
    def eligible(s, role):
        if role == 'acs_msun' or role == 'acs_mf':
            return s.get('can_acs', False)
        if role == 'mcnair':
            return s.get('covers_mcnair', False)
        if role == 'tsicu':
            return s.get('covers_tsicu', False)
        if role == 'sicu':
            return s.get('covers_sicu', False)
        if role == 'call':
            return s.get('can_call', False)
        return False

    model = cp_model.CpModel()

    # Variables: weekly assignments
    # acs_msun[w][s] = 1 if surgeon s does ACS M-Sun in week w
    acs_msun = [[model.NewBoolVar(f'acs_msun_w{w}_s{s}') 
                 for s in range(num_surgeons)] for w in range(num_weeks)]
    acs_mf = [[model.NewBoolVar(f'acs_mf_w{w}_s{s}') 
               for s in range(num_surgeons)] for w in range(num_weeks)]
    mcnair = [[model.NewBoolVar(f'mcnair_w{w}_s{s}') 
               for s in range(num_surgeons)] for w in range(num_weeks)]
    tsicu = [[model.NewBoolVar(f'tsicu_w{w}_s{s}') 
              for s in range(num_surgeons)] for w in range(num_weeks)]
    sicu = [[model.NewBoolVar(f'sicu_w{w}_s{s}') 
             for s in range(num_surgeons)] for w in range(num_weeks)]

    # Variables: nightly call
    call = [[model.NewBoolVar(f'call_d{d}_s{s}') 
             for s in range(num_surgeons)] for d in range(num_days)]

    # ─── HARD CONSTRAINTS ───────────────────────────────────────

    # 1. Each week needs exactly 1 per role
    for w in range(num_weeks):
        model.AddExactlyOne(acs_msun[w])
        model.AddExactlyOne(acs_mf[w])
        model.AddExactlyOne(mcnair[w])
        model.AddExactlyOne(tsicu[w])
        model.AddExactlyOne(sicu[w])

    # 2. Each night needs exactly 1 call surgeon
    for d in range(num_days):
        model.AddExactlyOne(call[d])

    # 3. Eligibility
    for w in range(num_weeks):
        for s in range(num_surgeons):
            if not eligible(surgeons[s], 'acs_msun'):
                model.Add(acs_msun[w][s] == 0)
            if not eligible(surgeons[s], 'acs_mf'):
                model.Add(acs_mf[w][s] == 0)
            if not eligible(surgeons[s], 'mcnair'):
                model.Add(mcnair[w][s] == 0)
            if not eligible(surgeons[s], 'tsicu'):
                model.Add(tsicu[w][s] == 0)
            if not eligible(surgeons[s], 'sicu'):
                model.Add(sicu[w][s] == 0)

    for d in range(num_days):
        for s in range(num_surgeons):
            if not eligible(surgeons[s], 'call'):
                model.Add(call[d][s] == 0)

    # 4. All 5 weekly roles must be different surgeons
    for w in range(num_weeks):
        for s in range(num_surgeons):
            # No surgeon can have more than 1 weekly role
            model.Add(
                acs_msun[w][s] + acs_mf[w][s] + mcnair[w][s] + 
                tsicu[w][s] + sicu[w][s] <= 1
            )

    # 5. No consecutive 7-day weeks (any two 7-day roles back to back)
    seven_day_roles = [acs_msun, mcnair, tsicu, sicu]
    for w in range(num_weeks - 1):
        for s in range(num_surgeons):
            for r1 in seven_day_roles:
                for r2 in seven_day_roles:
                    model.Add(r1[w][s] + r2[w+1][s] <= 1)

    # 6. ACS M-Sun cannot repeat following week
    for w in range(num_weeks - 1):
        for s in range(num_surgeons):
            model.Add(acs_msun[w][s] + acs_msun[w+1][s] <= 1)

    # 7. ICU surgeons: no call Mon-Sat that week
    # McNair: no call any night that week
    for w in range(num_weeks):
        week_start = weeks[w]['start']
        for d_offset in range(7):
            actual_date = week_start + timedelta(days=d_offset)
            if actual_date.month != month or actual_date.year != year:
                continue
            d = actual_date.day - 1  # 0-indexed
            if d >= num_days:
                continue
            day_of_week = actual_date.weekday()  # 0=Mon, 6=Sun
            
            for s in range(num_surgeons):
                # McNair: no call any night
                model.Add(mcnair[w][s] + call[d][s] <= 1)
                
                # TSICU/SICU: no call Mon-Sat (0-5)
                if day_of_week <= 5:
                    model.Add(tsicu[w][s] + call[d][s] <= 1)
                    model.Add(sicu[w][s] + call[d][s] <= 1)
                
                # ACS M-Sun: no call Mon-Sat
                if day_of_week <= 5:
                    model.Add(acs_msun[w][s] + call[d][s] <= 1)
                
                # ACS M-F: no call Mon-Thu (0-3)
                if day_of_week <= 3:
                    model.Add(acs_mf[w][s] + call[d][s] <= 1)

    # 8. Max call nights per month
    for s in range(num_surgeons):
        max_call = surgeons[s].get('max_call_per_month', 8)
        model.Add(sum(call[d][s] for d in range(num_days)) <= max_call)

    # 9. No more than 1 weekend call night per month per surgeon
    weekend_days = []
    for d in range(num_days):
        date = datetime(year, month, d + 1)
        if date.weekday() >= 5:  # Sat=5, Sun=6
            weekend_days.append(d)
    
    for s in range(num_surgeons):
        model.Add(sum(call[d][s] for d in weekend_days) <= 1)

    # ─── SOFT CONSTRAINTS / OBJECTIVES ──────────────────────────

    objective_terms = []

    # Balance FTE distribution
    target_shifts = {}
    for s in range(num_surgeons):
        fte = surgeons[s].get('fte', 1.0)
        annual_target = 168 * fte
        block_target = annual_target / 2
        prior = prior_totals.get(surgeons[s].get('name', ''), 0)
        remaining = max(0, annual_target - prior)
        target_shifts[s] = min(block_target, remaining)

    # Reward assignments that move surgeons toward their target
    for s in range(num_surgeons):
        target = target_shifts[s]
        if target <= 0:
            continue
        
        # ACS M-Sun = 7 shifts
        for w in range(num_weeks):
            if eligible(surgeons[s], 'acs_msun'):
                objective_terms.append(acs_msun[w][s])
        
        # ACS M-F = 5 shifts  
        for w in range(num_weeks):
            if eligible(surgeons[s], 'acs_mf'):
                objective_terms.append(acs_mf[w][s])

        # ICU = 7 shifts
        for w in range(num_weeks):
            if eligible(surgeons[s], 'mcnair'):
                objective_terms.append(mcnair[w][s])
            if eligible(surgeons[s], 'tsicu'):
                objective_terms.append(tsicu[w][s])
            if eligible(surgeons[s], 'sicu'):
                objective_terms.append(sicu[w][s])

    # Honor time off preferences
    pref_map = {}
    for p in preferences:
        sid = p.get('surgeon_id')
        pref_map[sid] = p

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 4

    if objective_terms:
        model.Maximize(sum(objective_terms))

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise Exception(f"No valid schedule found. Status: {solver.StatusName(status)}")

    # ─── BUILD RESULT ────────────────────────────────────────────

    result_weeks = []
    for w in range(num_weeks):
        week_data = {'label': weeks[w]['label']}
        for s in range(num_surgeons):
            name = surgeons[s]['name']
            if solver.Value(acs_msun[w][s]):
                week_data['ACS (M-Sun)'] = name
            if solver.Value(acs_mf[w][s]):
                week_data['ACS (M-F)'] = name
            if solver.Value(mcnair[w][s]):
                week_data['McNair ICU'] = name
            if solver.Value(tsicu[w][s]):
                week_data['TSICU'] = name
            if solver.Value(sicu[w][s]):
                week_data['SICU'] = name
        result_weeks.append(week_data)

    result_nights = {}
    for d in range(num_days):
        for s in range(num_surgeons):
            if solver.Value(call[d][s]):
                result_nights[str(d + 1)] = {
                    'Call': surgeons[s]['name'],
                    'Backup': ''
                }

    # Validation report
    violations = []
    warnings = []

    # Check all roles filled
    for w, week in enumerate(result_weeks):
        for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
            if role not in week:
                violations.append(f"Week {w+1}: {role} not assigned")

    for d in range(num_days):
        if str(d+1) not in result_nights:
            violations.append(f"Day {d+1}: No call surgeon assigned")

    return {
        'weeks': result_weeks,
        'nights': result_nights,
        'validation': {
            'violations': violations,
            'warnings': warnings,
            'valid': len(violations) == 0
        }
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
