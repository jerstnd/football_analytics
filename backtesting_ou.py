import os
import sys
import json
import argparse
import subprocess
import shutil
import numpy as np
import pandas as pd

# Import your Dixon-Coles math engine from poisson.py
from poisson import dc_pois, get_params

def calculate_ou_odds(matrix, line=2.5):
    """
    Sums the exact score probabilities for Over and Under a specific goal line.
    """
    prob_under = 0.0
    rows, cols = matrix.shape
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) < line:
                prob_under += matrix[i, j]
                
    prob_over = 1.0 - prob_under
    
    return {
        'Over': round(prob_over, 4),
        'Under': round(prob_under, 4)
    }

def get_true_ou_outcome(home_goals, away_goals, line=2.5):
    """Returns 'Over' or 'Under' based on actual total goals scored."""
    return 'Over' if (home_goals + away_goals) > line else 'Under'

def compute_ou_metrics(prob_dict, true_outcome, eps=1e-15):
    """
    Calculates Binary Log Loss and Brier Score for Over/Under.
    """
    y_true = 1.0 if true_outcome == 'Over' else 0.0
    p_over = np.clip(prob_dict['Over'], eps, 1.0 - eps)
    
    # Binary Log Loss
    log_loss = -(y_true * np.log(p_over) + (1 - y_true) * np.log(1 - p_over))
    
    # Binary Brier Score
    brier_score = (p_over - y_true) ** 2
    
    pred_outcome = 'Over' if p_over >= 0.5 else 'Under'
    is_correct = int(pred_outcome == true_outcome)
    
    return log_loss, brier_score, is_correct, pred_outcome

def run_weekly_update_script(w_bhm_path, weekly_csv, prior_state, post_state):
    """
    Invokes w_bhm.py via subprocess using 2 positional arguments.
    """
    # 1. Pass the PRIOR state so w_bhm.py reads it and generates prior_state_updated.json
    cmd = [
        sys.executable,
        w_bhm_path,
        weekly_csv,
        prior_state  
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[Error] w_bhm.py failed during weekly update:")
        if result.stdout.strip():
            print(f"STDOUT:\n{result.stdout.strip()}")
        if result.stderr.strip():
            print(f"STDERR:\n{result.stderr.strip()}")
        sys.exit(1)

    # 2. Find the file w_bhm.py just created based on the prior_state's name
    base_name = os.path.basename(prior_state).replace('.json', '')
    generated_file = os.path.join('state', f"{base_name}_updated.json")
    
    # 3. Rename it to become this gameweek's post_state (e.g., state/ratings_wk1.json)
    if os.path.exists(generated_file):
        if os.path.exists(post_state):
            os.remove(post_state)
        os.rename(generated_file, post_state)
    else:
        print(f"[Error] Expected w_bhm.py to create '{generated_file}', but file was not found.")
        sys.exit(1)

def run_ou_backtest(matches_csv, initial_state, w_bhm_path="w_bhm.py", rho=-0.10, line=2.5):
    if not os.path.exists(matches_csv):
        print(f"[Error] Could not find match data: {matches_csv}")
        sys.exit(1)
    if not os.path.exists(initial_state):
        print(f"[Error] Could not find initial state file: {initial_state}")
        sys.exit(1)

    df_matches = pd.read_csv(matches_csv)
    os.makedirs("prediction", exist_ok=True)
    
    wk_col = next((col for col in ['Gameweek', 'Week', 'Wk', 'gameweek', 'week'] if col in df_matches.columns), None)
    if wk_col:
        grouped_weeks = df_matches.groupby(wk_col, sort=True)
    else:
        df_matches['Auto_GW'] = (np.arange(len(df_matches)) // 10) + 1
        grouped_weeks = df_matches.groupby('Auto_GW', sort=True)

    all_results = []
    current_state_file = initial_state

    print("\n" + "="*80)
    print(f"      STARTING O/U {line} WALK-FORWARD BACKTEST")
    print("="*80)
    print(f"{'GW':<5} {'Matches':<9} {'Log Loss':<12} {'Brier Score':<14} {'Accuracy':<10} {'State Updated To'}")
    print("-" * 80)

    for gw, gw_df in grouped_weeks:
        with open(current_state_file, 'r') as f:
            state = json.load(f)
            
        name_to_id = {info['name']: str(t_id) for t_id, info in state['teams'].items()}
        id_to_name = {str(t_id): info['name'] for t_id, info in state['teams'].items()}
        
        gw_log_losses = []
        gw_brier_scores = []
        gw_correct = 0
        gw_count = 0
        
        for _, row in gw_df.iterrows():
            if 'home_id' in row and 'away_id' in row:
                h_id = str(int(row['home_id']))
                a_id = str(int(row['away_id']))
                if h_id not in id_to_name or a_id not in id_to_name: continue
                home_name = id_to_name[h_id]
                away_name = id_to_name[a_id]
            elif 'HomeTeam' in row and 'AwayTeam' in row:
                home_name = row['HomeTeam']
                away_name = row['AwayTeam']
                if home_name not in name_to_id or away_name not in name_to_id: continue
                h_id = name_to_id[home_name]
                a_id = name_to_id[away_name]

            home_goals = int(row['home_goals']) if 'home_goals' in row else int(row['FTHG'])
            away_goals = int(row['away_goals']) if 'away_goals' in row else int(row['FTAG'])

            # 1. Predict
            _, _, lambda_h, lambda_a = get_params(state, h_id, a_id)
            matrix = dc_pois(lambda_h, lambda_a, rho)
            
            # 2. Calculate O/U Odds instead of 1X2 Odds
            ou_odds = calculate_ou_odds(matrix, line=line)
            
            # 3. Evaluate
            true_outcome = get_true_ou_outcome(home_goals, away_goals, line=line)
            log_loss, brier_score, is_correct, pred_outcome = compute_ou_metrics(ou_odds, true_outcome)

            gw_log_losses.append(log_loss)
            gw_brier_scores.append(brier_score)
            gw_correct += is_correct
            gw_count += 1

            all_results.append({
                "Gameweek": gw,
                "Match": f"{home_name} vs {away_name}",
                "Score": f"{home_goals}-{away_goals}",
                "Total Goals": home_goals + away_goals,
                "True O/U": true_outcome,
                "Pred O/U": pred_outcome,
                "Over %": f"{ou_odds['Over']*100:.1f}%",
                "Under %": f"{ou_odds['Under']*100:.1f}%",
                "Log Loss": round(log_loss, 4),
                "Brier Score": round(brier_score, 4)
            })

        if gw_count == 0: continue

        gw_mean_ll = np.mean(gw_log_losses)
        gw_mean_bs = np.mean(gw_brier_scores)
        gw_acc = (gw_correct / gw_count) * 100.0

        temp_weekly_csv = f"data/temp_matches_gw{gw}.csv"
        gw_df.to_csv(temp_weekly_csv, index=False)
        next_state_file = f"state/ratings_wk{gw}.json"

        run_weekly_update_script(w_bhm_path, temp_weekly_csv, current_state_file, next_state_file)
        current_state_file = next_state_file

        print(f"GW {gw:<3} {gw_count:<9} {gw_mean_ll:<12.4f} {gw_mean_bs:<14.4f} {gw_acc:<8.1f}% -> {next_state_file}")

    df_all = pd.DataFrame(all_results)
    output_filepath = f"prediction/backtest_ou{str(line).replace('.','_')}_results.csv"
    df_all.to_csv(output_filepath, index=False)

    overall_ll = df_all['Log Loss'].mean()
    overall_bs = df_all['Brier Score'].mean()
    overall_acc = (df_all['True O/U'] == df_all['Pred O/U']).mean() * 100.0

    print("="*80)
    print(f"                 O/U {line} CUMULATIVE PERFORMANCE")
    print("="*80)
    print(f"Total Matches Evaluated : {len(df_all)}")
    print(f"Overall Top-Pick Acc    : {overall_acc:.2f}%")
    print("-" * 80)
    print(f"Overall Mean Log Loss   : {overall_ll:.4f}")
    print(f"Overall Mean Brier Score: {overall_bs:.4f}")
    print("="*80)
    print(f"Detailed match-by-match results saved to: {output_filepath}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Over/Under Backtester")
    parser.add_argument("--matches", required=True, help="Path to dataset")
    parser.add_argument("--initial_state", required=True, help="Path to initial state")
    parser.add_argument("--w_bhm", default="w_bhm.py", help="Path to your weekly MCMC update script")
    parser.add_argument("--rho", type=float, default=-0.10, help="Dixon-Coles parameter")
    parser.add_argument("--line", type=float, default=2.5, help="Over/Under goal line (default 2.5)")

    args = parser.parse_args()
    run_ou_backtest(args.matches, args.initial_state, args.w_bhm, args.rho, args.line)