import os
import sys
import json
import math
import argparse
import subprocess
import numpy as np
import pandas as pd
import shutil

# Import your Dixon-Coles math engine from poisson.py
from poisson import dc_pois, calculate_1x2_odds, get_params

def get_true_outcome(home_goals, away_goals):
    """Returns 'H', 'D', or 'A' based on actual goals scored."""
    if home_goals > away_goals:
        return 'H'
    elif home_goals == away_goals:
        return 'D'
    else:
        return 'A'

def compute_match_metrics(prob_dict, true_outcome, eps=1e-15):
    """
    Calculates single-match Log Loss and Brier Score.
    
    prob_dict: {'Home_Win': p_h, 'Draw': p_d, 'Away_Win': p_a}
    true_outcome: 'H', 'D', or 'A'
    """
    y_vec = np.array([
        1.0 if true_outcome == 'H' else 0.0,
        1.0 if true_outcome == 'D' else 0.0,
        1.0 if true_outcome == 'A' else 0.0
    ])
    
    p_vec = np.array([
        prob_dict['Home_Win'],
        prob_dict['Draw'],
        prob_dict['Away_Win']
    ], dtype=float)
    
    p_vec = np.clip(p_vec, eps, 1.0 - eps)
    p_vec = p_vec / np.sum(p_vec)
    
    log_loss = -np.sum(y_vec * np.log(p_vec))
    brier_score = np.sum((p_vec - y_vec) ** 2)

    cum_p = np.cumsum(p_vec)
    cum_y = np.cumsum(y_vec)
    rps = 0.5 * np.sum((cum_p[:-1] - cum_y[:-1]) ** 2)
    
    pred_outcome = ['H', 'D', 'A'][np.argmax(p_vec)]
    is_correct = int(pred_outcome == true_outcome)
    
    return log_loss, brier_score, rps, is_correct, pred_outcome

def run_weekly_update_script(w_bhm_path, weekly_csv, prior_state, post_state):
    """
    Invokes w_bhm.py via subprocess to update Bayesian parameters after each gameweek.
    Adjust the command list below if your w_bhm.py CLI expects different flag names.
    """
    cmd = [
        sys.executable,
        w_bhm_path,
        weekly_csv,   # sys.argv[1]: <new_matches.csv>
        prior_state   # sys.argv[2]: <previous_ratings.json> (e.g. state/ratings_wk0.json)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[Error] w_bhm.py failed during weekly update:")
        if result.stdout.strip():
            print(f"STDOUT:\n{result.stdout.strip()}")
        if result.stderr.strip():
            print(f"STDERR:\n{result.stderr.strip()}")
        sys.exit(1)

    # --- NEW: Find the '_updated.json' file w_bhm.py created and rename it to post_state ---
    base_name = os.path.basename(prior_state).replace('.json', '')
    generated_file = os.path.join('state', f"{base_name}_updated.json")
    
    if os.path.exists(generated_file):
        # Rename e.g. 'state/ratings_wk0_updated.json' -> 'state/ratings_wk1.json'
        if os.path.exists(post_state):
            os.remove(post_state) # Remove old file if overwriting
        os.rename(generated_file, post_state)
    else:
        print(f"[Error] Expected w_bhm.py to create '{generated_file}', but file was not found.")
        sys.exit(1)

def run_chronological_backtest(matches_csv, initial_state, w_bhm_path="w_bhm.py", rho=-0.10):
    if not os.path.exists(matches_csv):
        print(f"[Error] Could not find match data: {matches_csv}")
        sys.exit(1)
    if not os.path.exists(initial_state):
        print(f"[Error] Could not find initial state file: {initial_state}")
        sys.exit(1)

    df_matches = pd.read_csv(matches_csv)
    
    # Ensure directories exist
    os.makedirs("prediction", exist_ok=True)
    os.makedirs("state", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Determine Gameweek grouping
    # Checks if 'Gameweek', 'Week', or 'Wk' exists; otherwise chunks by 10 matches (38 weeks x 10 = 380 games)
    wk_col = next((col for col in ['Gameweek', 'Week', 'Wk', 'gameweek', 'week'] if col in df_matches.columns), None)
    if wk_col:
        print(f"Grouped season by column: '{wk_col}'")
        grouped_weeks = df_matches.groupby(wk_col, sort=True)
    else:
        print("No Gameweek column detected. Automatically chunking matches into 10-game Gameweeks...")
        df_matches['Auto_GW'] = (np.arange(len(df_matches)) // 10) + 1
        grouped_weeks = df_matches.groupby('Auto_GW', sort=True)

    all_results = []
    weekly_summaries = []

    # Initialize state pointer to Week 0 (from y_bhm.py)
    current_state_file = initial_state

    print("\n" + "="*80)
    print("      STARTING 38-WEEK CHRONOLOGICAL WALK-FORWARD BACKTEST")
    print("="*80)
    print(f"{'GW':<5} {'Matches':<9} {'Log Loss':<12} {'Brier Score':<14} {'Accuracy':<10} {'State Updated To'}")
    print("-" * 80)

    # 2. Chronological Gameweek Loop (Weeks 1 through 38)
    for gw, gw_df in grouped_weeks:
        with open(current_state_file, 'r') as f:
            state = json.load(f)
            
        name_to_id = {info['name']: str(t_id) for t_id, info in state['teams'].items()}
        id_to_name = {str(t_id): info['name'] for t_id, info in state['teams'].items()}
        
        gw_log_losses = []
        gw_brier_scores = []
        gw_rps_scores = []
        gw_correct = 0
        gw_count = 0
        
        for _, row in gw_df.iterrows():
            if 'home_id' in row and 'away_id' in row:
                h_id = str(int(row['home_id']))
                a_id = str(int(row['away_id']))
                if h_id not in id_to_name or a_id not in id_to_name:
                    continue
                home_name = id_to_name[h_id]
                away_name = id_to_name[a_id]
            elif 'HomeTeam' in row and 'AwayTeam' in row:
                home_name = row['HomeTeam']
                away_name = row['AwayTeam']
                if home_name not in name_to_id or away_name not in name_to_id:
                    continue
                h_id = name_to_id[home_name]
                a_id = name_to_id[away_name]
            else:
                print("[Error] CSV must contain either ('home_id', 'away_id') or ('HomeTeam', 'AwayTeam') columns.")
                sys.exit(1)

            home_goals = int(row['home_goals']) if 'home_goals' in row else int(row['FTHG'])
            away_goals = int(row['away_goals']) if 'away_goals' in row else int(row['FTAG'])

            # Predict using existing state
            _, _, lambda_h, lambda_a = get_params(state, h_id, a_id)
            matrix = dc_pois(lambda_h, lambda_a, rho)
            odds = calculate_1x2_odds(matrix)

            # Evaluate against true outcome
            true_outcome = get_true_outcome(home_goals, away_goals)
            log_loss, brier_score, match_rps, is_correct, pred_outcome = compute_match_metrics(odds, true_outcome)

            gw_log_losses.append(log_loss)
            gw_brier_scores.append(brier_score)
            gw_rps_scores.append(match_rps)
            gw_correct += is_correct
            gw_count += 1

            all_results.append({
                "Gameweek": gw,
                "Home Team": home_name,
                "Away Team": away_name,
                "Score": f"{home_goals}-{away_goals}",
                "True Result": true_outcome,
                "Pred Result": pred_outcome,
                "Home Win %": f"{odds['Home_Win']*100:.1f}%",
                "Draw %": f"{odds['Draw']*100:.1f}%",
                "Away Win %": f"{odds['Away_Win']*100:.1f}%",
                "Log Loss": round(log_loss, 4),
                "Brier Score": round(brier_score, 4),
                "RPS": round(match_rps, 4),
            })

        if gw_count == 0:
            continue

        gw_mean_ll = np.mean(gw_log_losses)
        gw_mean_bs = np.mean(gw_brier_scores)
        gw_acc = (gw_correct / gw_count) * 100.0

        # 3. Save this Gameweek's matches to a temporary file for w_bhm.py
        temp_weekly_csv = f"data/temp_matches_gw{gw}.csv"
        gw_df.to_csv(temp_weekly_csv, index=False)

        # Define next week's state filename
        next_state_file = f"state/ratings_wk{gw}.json"

        # Call w_bhm.py to perform the weekly MCMC update
        run_weekly_update_script(w_bhm_path, temp_weekly_csv, current_state_file, next_state_file)

        # Advance pointer for next week
        current_state_file = next_state_file

        print(f"GW {gw:<3} {gw_count:<9} {gw_mean_ll:<12.4f} {gw_mean_bs:<14.4f} {gw_acc:<8.1f}% -> {next_state_file}")

        weekly_summaries.append({
            "Gameweek": gw,
            "Matches": gw_count,
            "Log Loss": gw_mean_ll,
            "Brier Score": gw_mean_bs,
            "Accuracy %": gw_acc
        })

    # 4. Final Aggregations & Dashboard
    df_all = pd.DataFrame(all_results)
    output_filepath = "prediction/backtest_2526_results.csv"
    df_all.to_csv(output_filepath, index=False)

    overall_ll = df_all['Log Loss'].mean()
    overall_bs = df_all['Brier Score'].mean()
    overall_rps = df_all['RPS'].mean()
    overall_acc = (df_all['True Result'] == df_all['Pred Result']).mean() * 100.0

    print("="*80)
    print("                     SEASON CUMULATIVE PERFORMANCE")
    print("="*80)
    print(f"Total Matches Evaluated : {len(df_all)}")
    print(f"Overall Top-Pick Acc    : {overall_acc:.2f}%")
    print("-" * 80)
    print(f"Overall Mean Log Loss   : {overall_ll:.4f}")
    print(f"Overall Mean Brier Score: {overall_bs:.4f}")
    print(f"Overall Mean RPS        : {overall_rps:.4f}")
    print("="*80)
    print(f"Detailed match-by-match results saved to: {output_filepath}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronological 38-Week Walk-Forward Backtester")
    parser.add_argument("--matches", required=True, help="Path to 2025/26 dataset (e.g., data/matches_2526_ready.csv)")
    parser.add_argument("--initial_state", required=True, help="Path to initial state from y_bhm.py (e.g., state/ratings_wk0.json)")
    parser.add_argument("--w_bhm", default="w_bhm.py", help="Path to your weekly MCMC update script (default: w_bhm.py)")
    parser.add_argument("--rho", type=float, default=-0.10, help="Dixon-Coles correlation parameter")

    args = parser.parse_args()
    run_chronological_backtest(args.matches, args.initial_state, args.w_bhm, args.rho)