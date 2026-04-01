from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.sat.python import cp_model
import json
from datetime import datetime, timedelta
from calendar import monthrange

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
        block_number = data.get('block_number', 1)  # 1 = Jul-Dec, 2 = Jan-Jun
        
        result = solve_month(surgeons, year, month, preferences, prior_totals, block_number)
        return jsonify({'success': True, 'schedule': result})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

def get_weeks(year, month):
    days_in_month = monthrange(year, month)[1]
    first_day = datetime(year, month, 1)
    
    # Find first Monday on or before the 1st
    dow = first_day.weekday()
    week_start = first_day - timedelta(days=dow)
    
    weeks = []
    while True:
        week_end = week_start + timedelta(days=6)
        # Only include weeks that overlap with this month
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

    # ─── FTE CALCULATION ─────────────────────────────────────────
    # Annual target = 168 * FTE
    # Block 1 (Jul-Dec): target = 84 * FTE, prior = 0
    # Block 2 (Jan-Jun): target = annual - block1_actual
    # Per month target = block_target / 6
    def get_block_target(s):
        fte = s.get('fte', 1.0)
        annual_target = 168 * fte
        if block_number == 1:
            block_target = annual_target / 2
        else:
            prior = prior_totals.get(s.get('name', ''), 0)
            block_target = max(0, annual_target - prior)
        return block_target / 6  # per month target

    # ─── ELIGIBILITY ─────────────────────────────────────────────
    def eligible(s, role):
        if role == 'acs_msun' or role == 'acs_mf':
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

    # Identify fellows
    fellow_indices = [s for s in range(num_surgeons)
                      if 'fellow' in surgeons[s].get('name', '').lower() or
                         'Fellow' in surgeons[s].get('name', '')]

    model = cp_model.CpModel()

    # ─── VARIABLES ───────────────────────────────────────────────
    acs_msun = [[model.NewBoolVar(f'acs_msun_w{w}_s{s}')
                 for s in range(num_surgeons)] for w in range(num_weeks)]
    acs_mf   = [[model.NewBoolVar(f'acs_mf_w{w}_s{s}')
                 for s in range(num_surgeons)] for w in range(num_weeks)]
    mcnair   = [[model.NewBoolVar(f'mcnair_w{w}_s{s}')
                 for s in range(num_surgeons)] for w in range(num_weeks)]
    tsicu    = [[model.NewBoolVar(f'tsicu_w{w}_s{s}')
                 for s in range(num_surgeons)] for w in range(num_weeks)]
    sicu     = [[model.NewBoolVar(f'sicu_w{w}_s{s}')
                 for s in range(num_surgeons)] for w in range(num_weeks)]
    call     = [[model.NewBoolVar(f'call_d{d}_s{s}')
                 for s in range(num_surgeons)] for d in range(num_days)]

    # ─── HARD CONSTRAINTS ────────────────────────────────────────

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

    # 3. Eligibility — zero out ineligible assignments
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

    # 4. No surgeon can hold more than 1 weekly role per week
    for w in range(num_weeks):
        for s in range(num_surgeons):
            model.Add(
                acs_msun[w][s] + acs_mf[w][s] + mcnair[w][s] +
                tsicu[w][s] + sicu[w][s] <= 1
            )

    # 5. No two consecutive 7-day service weeks for same surgeon
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

    # 7. ACS call restrictions and ICU call restrictions
    for w in range(num_weeks):
        week_start = weeks[w]['start']
        for d_offset in range(7):
            actual_date = week_start + timedelta(days=d_offset)
            if actual_date.month != month or actual_date.year != year:
                continue
            d = actual_date.day - 1
            if d >= num_days:
                continue
            day_of_week = actual_date.weekday()  # 0=Mon, 6=Sun

            for s in range(num_surgeons):
                # McNair: no call ANY night that week
                model.Add(mcnair[w][s] + call[d][s] <= 1)

                # TSICU/SICU: no call Mon-Sat (last resort Sun only)
                if day_of_week <= 5:
                    model.Add(tsicu[w][s] + call[d][s] <= 1)
                    model.Add(sicu[w][s] + call[d][s] <= 1)

                # ACS M-Sun: no call Mon-Sat
                if day_of_week <= 5:
                    model.Add(acs_msun[w][s] + call[d][s] <= 1)

                # ACS M-F: no call Mon-Thu
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
        if date.weekday() >= 5:
            weekend_days.append(d)
    for s in range(num_surgeons):
        if weekend_days:
            model.Add(sum(call[d][s] for d in weekend_days) <= 1)

    # 10. Fellows cannot be on same role in same week
    if len(fellow_indices) >= 2:
        for w in range(num_weeks):
            for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                model.Add(sum(role[w][f] for f in fellow_indices) <= 1)

    # 11. No surgeon assigned ACS M-F more than 2 weeks per month
    for s in range(num_surgeons):
        model.Add(sum(acs_mf[w][s] for w in range(num_weeks)) <= 2)

    # 12. No surgeon assigned ACS M-Sun more than 2 weeks per month
    for s in range(num_surgeons):
        model.Add(sum(acs_msun[w][s] for w in range(num_weeks)) <= 2)

    # ─── SOFT OBJECTIVES ─────────────────────────────────────────
    objective_terms = []
    penalty_terms = []

    for s in range(num_surgeons):
        monthly_target = get_block_target(surgeons[s])
        pref = surgeons[s].get('extra_shift_preference', 'baseline')

        # Reward service week assignments weighted by FTE target
        weight = max(1, int(monthly_target))

        for w in range(num_weeks):
            if eligible(surgeons[s], 'acs_msun'):
                objective_terms.append((weight, acs_msun[w][s]))
            if eligible(surgeons[s], 'acs_mf'):
                objective_terms.append((weight, acs_mf[w][s]))
            if eligible(surgeons[s], 'mcnair'):
                objective_terms.append((weight, mcnair[w][s]))
            if eligible(surgeons[s], 'tsicu'):
                objective_terms.append((weight, tsicu[w][s]))
            if eligible(surgeons[s], 'sicu'):
                objective_terms.append((weight, sicu[w][s]))

        # Penalize over-assignment for baseline-only surgeons
        if pref == 'baseline':
            for w in range(num_weeks):
                for role in [acs_msun, acs_mf, mcnair, tsicu, sicu]:
                    penalty_terms.append(role[w][s])

    # Penalize Rojas-Khalil call on non-Fri/Sat
    rojas_idx = next((s for s in range(num_surgeons)
                      if 'Rojas' in surgeons[s].get('name', '')), None)
    if rojas_idx is not None:
        for d in range(num_days):
            date = datetime(year, month, d + 1)
            dow = date.weekday()
            if dow not in [4, 5]:  # Not Fri or Sat
                penalty_terms.append(call[d][rojas_idx])

    # Build objective
    if objective_terms:
        model.Maximize(
            sum(w * v for w, v in objective_terms) -
            sum(penalty_terms) * 10
        )

    # ─── SOLVE ───────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 4

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise Exception(
            f"No valid schedule found for {year}-{month:02d}. "
            f"Status: {solver.StatusName(status)}. "
            f"Check surgeon eligibility — you may not have enough surgeons for all roles."
        )

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

    # ─── VALIDATION REPORT ───────────────────────────────────────
    violations = []
    warnings = []

    for w, week in enumerate(result_weeks):
        for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
            if role not in week:
                violations.append(f"Week {w+1}: {role} not assigned")

    for d in range(num_days):
        if str(d+1) not in result_nights:
            violations.append(f"Day {d+1}: No call surgeon assigned")

    # Check Perez not over-assigned
    perez_mf = sum(1 for w in result_weeks if w.get('ACS (M-F)') and
                   'Perez' in w.get('ACS (M-F)', ''))
    if perez_mf > 2:
        warnings.append(f"Perez assigned ACS M-F {perez_mf} weeks — check distribution")

    # Check fellows not doubled up
    for w, week in enumerate(result_weeks):
        fellow_assignments = []
        for role in ['ACS (M-Sun)', 'ACS (M-F)', 'McNair ICU', 'TSICU', 'SICU']:
            name = week.get(role, '')
            if 'Fellow' in name or 'fellow' in name:
                fellow_assignments.append(name)
        if len(set(fellow_assignments)) < len(fellow_assignments):
            violations.append(f"Week {w+1}: Same fellow assigned to multiple roles")

    # FTE distribution summary
    fte_summary = {}
    for s in range(num_surgeons):
        name = surgeons[s]['name']
        shifts = 0
        for w in result_weeks:
            if w.get('ACS (M-F)') == name:
                shifts += 5
            if w.get('ACS (M-Sun)') == name:
                shifts += 7
            for role in ['McNair ICU', 'TSICU', 'SICU']:
                if w.get(role) == name:
                    shifts += 7
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
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
