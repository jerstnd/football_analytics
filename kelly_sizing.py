import sys
import pandas as pd
import argparse

import db

def calculate_kelly_orders(df, bankroll, fraction=0.25):
    '''
    df needs columns: match_id, Home Team, Away Team, Prob_Over_25, Prob_Under_25,
    Odds_Over_25, Odds_Under_25 (odds of 0 mean "no odds", which never bets).
    Returns the orders as a DataFrame, largest wager first.
    '''
    orders = []

    for index, row in df.iterrows():
        match_name = f"{row['Home Team']} vs {row['Away Team']}"
        
        # 1. Over 2.5 Kelly Math
        ev_over = -1.0
        kelly_over = 0.0
        if row['Odds_Over_25'] > 0:
            ev_over = (row['Prob_Over_25'] * row['Odds_Over_25']) - 1.0
            if ev_over > 0:
                b_over = row['Odds_Over_25'] - 1.0
                kelly_over = ((row['Prob_Over_25'] * b_over) - (1.0 - row['Prob_Over_25'])) / b_over

        # 2. Under 2.5 Kelly Math
        ev_under = -1.0
        kelly_under = 0.0
        if row['Odds_Under_25'] > 0:
            ev_under = (row['Prob_Under_25'] * row['Odds_Under_25']) - 1.0
            if ev_under > 0:
                b_under = row['Odds_Under_25'] - 1.0
                kelly_under = ((row['Prob_Under_25'] * b_under) - (1.0 - row['Prob_Under_25'])) / b_under

        # 3. Find the best actionable bet for the match
        action = "No Bet"
        final_ev = 0.0
        final_kelly = 0.0
        book_odds = 0.0
        model_prob = 0.0
        
        if ev_over > 0 and ev_over > ev_under:
            action = "Over 2.5"
            final_ev = ev_over
            final_kelly = kelly_over
            book_odds = row['Odds_Over_25']
            model_prob = row['Prob_Over_25']
        elif ev_under > 0 and ev_under > ev_over:
            action = "Under 2.5"
            final_ev = ev_under
            final_kelly = kelly_under
            book_odds = row['Odds_Under_25']
            model_prob = row['Prob_Under_25']

        adj_kelly = final_kelly * fraction
        position_size = adj_kelly * bankroll
        
        if position_size > 0:
            orders.append({
                "match_id": row['match_id'],
                "ev": final_ev,
                "adj_kelly": adj_kelly,
                "Match": match_name,
                "Action": action,
                "Prob": round(model_prob, 4),
                "Odds": book_odds,
                "Edge (EV)": f"+{final_ev * 100:.1f}%",
                "Adj_Kelly": f"{adj_kelly * 100:.2f}%",
                "Wager_$": round(position_size, 2)
            })

    df_orders = pd.DataFrame(orders)

    print("\n" + "="*80)
    print(f" ACTIONABLE TRADES (Bankroll: ${bankroll:,.2f} | Kelly Fraction: {fraction})")
    print("="*80)

    if df_orders.empty:
        print("No profitable +EV bets found. Keep capital protected.")
    else:
        df_orders = df_orders.sort_values(by='Wager_$', ascending=False)
        print(df_orders.drop(columns=['match_id', 'ev', 'adj_kelly']).to_string(index=False))

    print("="*80 + "\n")
    return df_orders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fractional Kelly Position Sizer")
    parser.add_argument("--season", required=True, help="Season code, e.g. 2627")
    parser.add_argument("--round", type=int, required=True, help="Round whose predictions to size")
    parser.add_argument("--bankroll", type=float, default=10000.0, help="Total available capital")
    parser.add_argument("--fraction", type=float, default=0.25, help="Fractional Kelly multiplier (default: 0.25)")
    args = parser.parse_args()

    conn = db.connect()
    rows = db.get_predictions(conn, args.season, args.round)
    if not rows:
        print(f"[Error] No predictions for season {args.season} round {args.round}. Run poisson.py first.")
        sys.exit(1)

    df = pd.DataFrame([{
        'match_id': r['match_id'], 'Home Team': r['home'], 'Away Team': r['away'],
        'Prob_Over_25': r['prob_over_25'], 'Prob_Under_25': r['prob_under_25'],
        'Odds_Over_25': r['odds_over_25'] or 0.0, 'Odds_Under_25': r['odds_under_25'] or 0.0,
    } for r in rows])
    missing = int((df['Odds_Over_25'] == 0).sum())
    if missing:
        print(f"[Notice] {missing} of {len(df)} matches have no O/U odds. Add them with fetch.py --odds.")

    df_orders = calculate_kelly_orders(df, args.bankroll, args.fraction)

    with conn:
        conn.execute('DELETE FROM orders WHERE match_id IN (SELECT id FROM matches WHERE season = ? AND round = ?)',
                     (args.season, args.round))
        conn.executemany('''INSERT INTO orders (match_id, action, prob, odds, ev, adj_kelly, stake, bankroll, fraction)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         [(int(o['match_id']), o['Action'], o['Prob'], o['Odds'], o['ev'], o['adj_kelly'],
                           o['Wager_$'], args.bankroll, args.fraction) for _, o in df_orders.iterrows()])
    print(f"Saved {len(df_orders)} orders: season {args.season}, round {args.round}")
