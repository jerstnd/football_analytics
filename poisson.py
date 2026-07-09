import sys
import json
import math
import numpy as np
import pandas as pd
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
    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python poisson.py <path_to_state.json> <home_team_id> <away_team_id>')
        sys.exit(1)

    json_file = sys.argv[1]
    home_team_id = str(sys.argv[2])
    away_team_id = str(sys.argv[3])

    with open(json_file, 'r') as f:
        current_state = json.load(f)

    h_name, a_name, lambda_h, lambda_a = get_params(current_state, home_team_id, away_team_id)

    RHO = -0.10
    matrix_out = dc_pois(lambda_h, lambda_a, RHO, max_goals=5)
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
    print(df_matrix.round(4) * 100) # Multiplied to show percentages
    
    print("\n--- FINAL 1X2 PROBABILITIES ---")
    print(f"Home Win: {final_odds['Home_Win'] * 100:.2f}%")
    print(f"Draw:     {final_odds['Draw'] * 100:.2f}%")
    print(f"Away Win: {final_odds['Away_Win'] * 100:.2f}%")