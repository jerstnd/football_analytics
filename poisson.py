import sys
import math
import numpy as np
import pandas as pd
import argparse
from scipy.stats import poisson

import db

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

def predict_batch(state, fixtures, rating_round, rho=-0.1285):
    '''
    Predicts every fixture (rows from db.get_fixtures) and returns prediction dicts
    ready for db.save_predictions.
    '''
    predictions = []
    for fx in fixtures:
        # Promoted teams must be initialised by y_bhm.py, which gives them a proper
        # prior. Silently treating them as average (0.0) teams is wrong.
        for tid, name in [(fx['home_id'], fx['home']), (fx['away_id'], fx['away'])]:
            if str(tid) not in state['teams']:
                print(f"[Error] '{name}' is not in the ratings. Re-run y_bhm.py to initialise promoted teams.")
                sys.exit(1)

        _, _, lambda_h, lambda_a = get_params(state, str(fx['home_id']), str(fx['away_id']))
        matrix = dc_pois(lambda_h, lambda_a, rho)
        ou_probs = calculate_ou_25(matrix)
        odds_1x2 = calculate_1x2_odds(matrix)

        predictions.append({
            'match_id': fx['id'],
            'rating_round': rating_round,
            'home': fx['home'],
            'away': fx['away'],
            'xg_h': round(lambda_h, 2),
            'xg_a': round(lambda_a, 2),
            'prob_h': odds_1x2['Home_Win'],
            'prob_d': odds_1x2['Draw'],
            'prob_a': odds_1x2['Away_Win'],
            'prob_over_25': ou_probs['Over_25'],
            'prob_under_25': ou_probs['Under_25'],
        })
    return predictions


def print_predictions(predictions, title):
    df = pd.DataFrame(predictions)[['home', 'away', 'xg_h', 'xg_a', 'prob_h', 'prob_d', 'prob_a',
                                    'prob_over_25', 'prob_under_25']]
    df.columns = ['Home Team', 'Away Team', 'xG_H', 'xG_A', 'Prob_H', 'Prob_D', 'Prob_A',
                  'Prob_Over_25', 'Prob_Under_25']
    print("\n" + "="*100)
    print(title.center(100))
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dixon-Coles Match Predictor")
    parser.add_argument("--season", required=True, help="Season code, e.g. 2627")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--round", type=int, help="Predict every fixture of this round (uses ratings of round-1)")
    group.add_argument("--match", nargs=2, metavar=('HOME', 'AWAY'),
                       help="Predict a single match (team names or IDs) with the latest ratings")
    args = parser.parse_args()

    conn = db.connect()

    if args.match:
        rating_round = db.latest_round(conn, args.season)
        state = db.load_state(conn, args.season, rating_round) if rating_round is not None else None
        if state is None:
            print(f"[Error] No ratings for season {args.season}.")
            sys.exit(1)
        ids = []
        for t in args.match:
            tid = int(t) if t.isdigit() else db.resolve_team(conn, t)
            if tid is None or str(tid) not in state['teams']:
                print(f"[Error] Team '{t}' not found in the season {args.season} ratings.")
                sys.exit(1)
            ids.append(str(tid))
        print(f"Using ratings after round {rating_round}")
        predict_single(state, ids[0], ids[1])
    else:
        state = db.load_state(conn, args.season, args.round - 1)
        if state is None:
            print(f"[Error] No ratings for season {args.season} round {args.round - 1}. Run w_bhm.py first.")
            sys.exit(1)
        fixtures = db.get_fixtures(conn, args.season, args.round)
        if not fixtures:
            print(f"[Error] No fixtures for season {args.season} round {args.round}. Run fetch.py first.")
            sys.exit(1)
        predictions = predict_batch(state, fixtures, args.round - 1)
        print_predictions(predictions, f"ROUND {args.round} PREDICTIONS (ratings after round {args.round - 1})")
        db.save_predictions(conn, predictions)
        print(f"Saved {len(predictions)} predictions: season {args.season}, round {args.round}")
