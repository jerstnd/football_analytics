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
    Extract the Home win, Draw Away win probabilities from DC Matrix
    '''

    home_win_prob = np.sum(np.tril(matrix, -1))
    draw_prob = np.sum(np.diag(matrix))
    away_win_prob = np.sum(np.triu(matrix, 1))

    return {
        'Home_Win': round(home_win_prob, 4),
        'Draw': round(draw_prob, 4),
        'Away_Win': round(away_win_prob, 4)
    }

def calculate_ou_25(matrix):
    '''
    Extracts Over/Under 2.5 goals probabilities from the DC Matrix
    '''
    # Create a boolean mask for coordinates where total goals < 3
    under_mask = np.array([[i + j < 3 for j in range(matrix.shape[1])] for i in range(matrix.shape[0])])
    
    under_prob = np.sum(matrix[under_mask])
    over_prob = 1.0 - under_prob
    
    return {
        'Over_25': round(over_prob, 4),
        'Under_25': round(under_prob, 4)
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

    INTERCEPT = 0.2231

    lambda_h = math.exp(INTERCEPT + h_adv + h_att + a_def)
    lambda_a = math.exp(INTERCEPT + a_att + h_def)

    return h_name, a_name, lambda_h, lambda_a

def predict_single(state, h_id, a_id, rho=-0.1285): # -0.1285 is a general rho number for premier league
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

def predict_batch(state, fixtures_csv, rho=-0.1285, state_filepath=None):
    '''
    Predicts a full weekend CSV and saves ALL probabilities to a summary DataFrame
    '''
    if not os.path.exists(fixtures_csv):
        print(f"[Error] Could not find {fixtures_csv}")
        sys.exit(1)
        
    name_to_id = {info['name']: str(t_id) for t_id, info in state['teams'].items()}
    fixtures = pd.read_csv(fixtures_csv)
    results = []
    
    state_updated = False
    
    for index, row in fixtures.iterrows():
        home_name = str(row['HomeTeam']).strip()
        away_name = str(row['AwayTeam']).strip()
        
        # PROMOTED TEAM INITIALIZATION LOGIC
        for team_name in [home_name, away_name]:
            if team_name not in name_to_id:
                existing_ids = [int(k) for k in state['teams'].keys()]
                new_id = str(max(existing_ids) + 1 if existing_ids else 1)
                
                state['teams'][new_id] = {'name': team_name, 'att_mean': 0.0, 'def_mean': 0.0}
                name_to_id[team_name] = new_id
                state_updated = True
                print(f"  [+] Initialized promoted team: {team_name} (ID: {new_id})")
            
        h_id = name_to_id[home_name]
        a_id = name_to_id[away_name]
        
        _, _, lambda_h, lambda_a = get_params(state, h_id, a_id)
        
        matrix = dc_pois(lambda_h, lambda_a, rho)
        
        # Calculate BOTH markets
        ou_probs = calculate_ou_25(matrix)
        odds_1x2 = calculate_1x2_odds(matrix)
        
        # Append ALL predictions
        results.append({
            "Home Team": home_name,
            "Away Team": away_name,
            "xG_H": round(lambda_h, 2),
            "xG_A": round(lambda_a, 2),
            "Prob_H": odds_1x2['Home_Win'],
            "Prob_D": odds_1x2['Draw'],
            "Prob_A": odds_1x2['Away_Win'],
            "Prob_Over_25": ou_probs['Over_25'],
            "Prob_Under_25": ou_probs['Under_25'],
            "Odds_Over_25": float(row.get('Odds_Over_25', 0.0)),
            "Odds_Under_25": float(row.get('Odds_Under_25', 0.0))
        })
        
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print(f" MATCHDAY PREDICTIONS SUMMARY".center(100))
    print("="*100)
    # Print the expanded view to the terminal
    display_cols = ['Home Team', 'Away Team', 'xG_H', 'xG_A', 'Prob_H', 'Prob_D', 'Prob_A', 'Prob_Over_25', 'Prob_Under_25']
    print(df_results[display_cols].to_string(index=False))
    print("="*100 + "\n")

    base_filename = os.path.basename(fixtures_csv)
    output_filename = base_filename.replace('.csv', '_predictions.csv')
    output_filepath = os.path.join('prediction', output_filename)

    # Save everything to the CSV so kelly_sizing.py can still read it
    df_results.to_csv(output_filepath, index=False)
    print(f"Saved Matchday Predictions to: {output_filepath}")

    # Save state if new teams were initialized
    if state_updated and state_filepath:
        with open(state_filepath, 'w') as f:
            json.dump(state, f, indent=4)
        print(f"Saved updated JSON state to: {state_filepath}")  


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
        predict_batch(current_state, args.batch, state_filepath=args.state)