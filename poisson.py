import os
import sys
import json
import math
import numpy as np
import pandas as pd
import argparse
from scipy.stats import poisson

def dc_pois(lambda_h: float, lambda_a: float, rho: float, max_goals: int = 5):
    '''
    Generates 2D probability matrix using Dixon-Coles poisson method
    '''

    # Init empty grid
    base_matrix = np.zeros((max_goals + 1, max_goals + 1))

    # Calculate poisson distribution probability 
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_h = poisson.pmf(i, lambda_h)
            prob_a = poisson.pmf(j, lambda_a)
            base_matrix[i,j] = prob_h * prob_a
    
    # Dixon-Coles adjustment grid (tau)
    tau = np.ones((max_goals + 1, max_goals + 1))

    tau[0, 0] = 1 - (lambda_h * lambda_a * rho)
    tau[1, 0] = 1 + (lambda_a * rho)
    tau[0, 1] = 1 + (lambda_h * rho)
    tau[1, 1] = 1 - rho

    dc_matrix = base_matrix * tau

    return dc_matrix

def calculate_1x2_odds(matrix):
    '''
    Extract the Home win, Draw, Away win probabilities from DC Matrix
    '''

    home_win_prob = np.sum(np.tril(matrix, -1))
    draw_prob = np.sum(np.diag(matrix))
    away_win_prob = np.sum(np.triu(matrix, 1))

    return {
        'Home_Win': round(home_win_prob, 4),
        'Draw': round(draw_prob, 4),
        'Away_Win': round(away_win_prob, 4)
    }

def get_params(state, h_id, a_id):
    '''
    Pulls MCMC params from JSON and calculate to Expected Goals (lambda)
    '''

    try:
        h_adv = state['home_adv']['mean']

        h_name = state['teams'][h_id]['name']
        h_att = state['teams'][h_id]['att_mean']
        h_def = state['teams'][h_id]['def_mean']

        a_name = state['teams'][a_id]['name']
        a_att = state['teams'][a_id]['att_mean']
        a_def = state['teams'][a_id]['def_mean']
    except KeyError as e:
        print(f"Error: Could not find team ID {e} in the JSON file.")
        sys.exit(1)

    lambda_h = math.exp(h_adv + h_att + h_def)
    lambda_a = math.exp(a_att + a_def)

    return h_name, a_name, lambda_h, lambda_a

def predict_single(state, h_id, a_id, rho=-0.10):
    '''
    Predicts and prints a single match layout
    '''
    h_name, a_name, lambda_h, lambda_a = get_params(state, h_id, a_id)

    matrix_out = dc_pois(lambda_h, lambda_a, rho, max_goals=5)
    final_odds = calculate_1x2_odds(matrix_out)

    df_matrix = pd.DataFrame(
        matrix_out, 
        index=[f"Home {i}" for i in range(6)],
        columns=[f"Away {i}" for i in range(6)]
    )

    print("\n" + "="*50)
    print(f" MATCH PREDICTION: {h_name} vs {a_name}")
    print("="*50)
    print(f"Expected Goals (xG):")
    print(f"  {h_name:<10}: {lambda_h:.3f}")
    print(f"  {a_name:<10}: {lambda_a:.3f}")

    print("--- EXACT SCORE PROBABILITY MATRIX (Dixon-Coles) ---")
    print(df_matrix.round(4) * 100)
    
    print("\n--- FINAL 1X2 PROBABILITIES ---")
    print(f"Home Win: {final_odds['Home_Win'] * 100:.2f}%")
    print(f"Draw:     {final_odds['Draw'] * 100:.2f}%")
    print(f"Away Win: {final_odds['Away_Win'] * 100:.2f}%")

# --- NEW: Added the batch matchday processor function ---
def predict_batch(state, fixtures_csv, rho=-0.10):
    '''
    Predicts a full weekend CSV and saves a summary DataFrame
    '''
    if not os.path.exists(fixtures_csv):
        print(f"[Error] Could not find {fixtures_csv}")
        sys.exit(1)
        
    # Dynamically build team mapping (Name -> ID) straight from your JSON state
    name_to_id = {info['name']: str(t_id) for t_id, info in state['teams'].items()}
    
    fixtures = pd.read_csv(fixtures_csv)
    results = []
    
    for index, row in fixtures.iterrows():
        home_name = row['HomeTeam']
        away_name = row['AwayTeam']
        
        if home_name not in name_to_id or away_name not in name_to_id:
            print(f"  [!] Skipping {home_name} vs {away_name}: Team not found in state JSON.")
            continue
            
        h_id = name_to_id[home_name]
        a_id = name_to_id[away_name]
        
        # Use your existing params function
        _, _, lambda_h, lambda_a = get_params(state, h_id, a_id)
        
        matrix = dc_pois(lambda_h, lambda_a, rho)
        odds = calculate_1x2_odds(matrix)
        
        results.append({
            "Home Team": home_name,
            "Away Team": away_name,
            "xG (H)": f"{lambda_h:.2f}",
            "xG (A)": f"{lambda_a:.2f}",
            "Home Win %": f"{odds['Home_Win'] * 100:.1f}%",
            "Draw %": f"{odds['Draw'] * 100:.1f}%",
            "Away Win %": f"{odds['Away_Win'] * 100:.1f}%"
        })
        
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*85)
    print(f" MATCHDAY PREDICTIONS SUMMARY")
    print("="*85)
    print(df_results.to_string(index=False))
    print("="*85 + "\n")
    
    output_filepath = fixtures_csv.replace('.csv', '_predictions.csv')
    df_results.to_csv(output_filepath, index=False)
    print(f"Saved Matchday Summary to: {output_filepath}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dixon-Coles Match Predictor")
    parser.add_argument("--state", required=True, help="Path to state JSON file")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--match", nargs=2, metavar=('HOME_ID', 'AWAY_ID'), help="Predict single match using team IDs")
    group.add_argument("--batch", help="Path to fixtures CSV for matchday summary")
    
    args = parser.parse_args()

    with open(args.state, 'r') as f:
        current_state = json.load(f)

    if args.match:
        predict_single(current_state, args.match[0], args.match[1])
    elif args.batch:
        predict_batch(current_state, args.batch)