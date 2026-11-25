'''
Walk-forward backtest of the full pipeline, read from and saved to football.db.

  python backtest/backtest.py --season 2526                 # fit on 2425, replay 2025/26
  python backtest/backtest.py --season 2425 2526 --seed 7   # several seasons
  python backtest/backtest.py --report 3                    # re-report a saved run
  python backtest/backtest.py --list                        # list saved runs

For each test season it fits the preseason prior on the previous season (y_bhm),
then steps through the season: predict every match in the next step with the
current ratings (poisson), then add those results to the ratings (w_bhm). This is
exactly the live weekly workflow, so nothing from the future reaches a prediction.

Steps follow match dates rather than official rounds: a postponed match is
predicted and learned from when it was actually played, not with its original round.

The report scores the model against Bet365's odds (margin removed) and simulates
betting at Bet365 and at the best available price.
'''
import os
import sys
import math
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db                                   # noqa: E402
from y_bhm import fit_preseason             # noqa: E402
from w_bhm import update_state              # noqa: E402
from poisson import predict_batch           # noqa: E402

RHO = -0.1285


# ---------- running ----------

def assign_steps(matches):
    '''
    Gives each match the update step it was played in. Step r opens on the median
    date of round r; a match belongs to the latest step opened on or before its date,
    and never to one earlier than its own round.
    '''
    by_round = {}
    for m in matches:
        by_round.setdefault(m['round'], []).append(m['date'])
    opens = {r: sorted(ds)[len(ds) // 2] for r, ds in by_round.items()}
    steps = {}
    for m in matches:
        later = [r for r, d in opens.items() if d <= m['date']]
        steps[m['id']] = max([m['round']] + later)
    return steps


def run_backtest(conn, season, train, rho, seed, note=None):
    random.seed(seed)
    id_to_name = db.team_names(conn)
    train_matches = [m for s in train for m in db.get_results(conn, s)]
    if not train_matches:
        raise SystemExit(f"[Error] No results for training season(s) {train}.")

    matches = conn.execute('''
        SELECT m.id, m.round, m.date, m.home_id, m.away_id, h.name AS home, a.name AS away,
               m.home_goals, m.away_goals, m.home_sot, m.away_sot
        FROM matches m JOIN teams h ON h.id = m.home_id JOIN teams a ON a.id = m.away_id
        WHERE m.season = ? AND m.home_goals IS NOT NULL''', (season,)).fetchall()
    if not matches or any(m['round'] is None or m['date'] is None for m in matches):
        raise SystemExit(f"[Error] Season {season} needs results with rounds and dates. Run fetch.py --season {season}.")

    steps = assign_steps(matches)
    moved = sum(steps[m['id']] != m['round'] for m in matches)
    print(f"\n=== Backtest {season} (trained on {', '.join(train)}, rho {rho}, seed {seed}) ===")
    print(f"{len(matches)} matches in {len(set(steps.values()))} steps; {moved} played late and moved to a later step.")

    state = fit_preseason(train_matches, db.season_team_ids(conn, season), id_to_name)

    with conn:
        run_id = conn.execute('INSERT INTO backtest_runs (season, train_seasons, rho, seed, note) VALUES (?, ?, ?, ?, ?)',
                              (season, ','.join(train), rho, seed, note)).lastrowid

    for step in sorted(set(steps.values())):
        fixtures = [dict(m) for m in matches if steps[m['id']] == step]
        preds = predict_batch(state, fixtures, rating_round=step - 1, rho=rho)
        with conn:
            conn.executemany('''
                INSERT INTO backtest_predictions (run_id, match_id, slot, xg_h, xg_a, prob_h, prob_d, prob_a,
                                                  prob_over_25, prob_under_25)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                [(run_id, p['match_id'], step, p['xg_h'], p['xg_a'], p['prob_h'], p['prob_d'], p['prob_a'],
                  p['prob_over_25'], p['prob_under_25']) for p in preds])
        results = [(m['home_id'], m['away_id'], db.blend(m['home_goals'], m['home_sot']),
                    db.blend(m['away_goals'], m['away_sot'])) for m in fixtures]
        state = update_state(state, results)
        print(f"  step {step:>2}: {len(fixtures):>2} matches predicted, ratings updated", flush=True)

    print(f"Saved as run {run_id}.")
    return run_id


# ---------- scoring ----------

def scores_1x2(p, outcome):
    '''p = (home, draw, away) summing to 1; outcome = 0/1/2. Returns log loss, RPS, Brier.'''
    y = [1.0 if k == outcome else 0.0 for k in range(3)]
    ll = -math.log(max(p[outcome], 1e-15))
    rps = ((p[0] - y[0]) ** 2 + (p[0] + p[1] - y[0] - y[1]) ** 2) / 2
    brier = sum((p[k] - y[k]) ** 2 for k in range(3))
    return ll, rps, brier


def no_margin(odds):
    inv = [1 / o for o in odds]
    return [v / sum(inv) for v in inv]


def load_run(conn, run_id):
    return conn.execute('''
        SELECT bp.*, m.round, m.date, m.home_goals, m.away_goals, h.name AS home, a.name AS away,
               b.home AS b_h, b.draw AS b_d, b.away AS b_a, b.over_25 AS b_o, b.under_25 AS b_u,
               x.home AS x_h, x.draw AS x_d, x.away AS x_a, x.over_25 AS x_o, x.under_25 AS x_u
        FROM backtest_predictions bp
        JOIN matches m ON m.id = bp.match_id
        JOIN teams h ON h.id = m.home_id JOIN teams a ON a.id = m.away_id
        LEFT JOIN odds b ON b.match_id = m.id AND b.source = 'bet365'
        LEFT JOIN odds x ON x.match_id = m.id AND x.source = 'best'
        WHERE bp.run_id = ? ORDER BY bp.slot, m.date, m.id''', (run_id,)).fetchall()


def simulate(bets, kelly_fraction, bankroll=1000.0):
    '''bets: (prob, odds, won) in date order. Flat 1-unit stakes and fractional Kelly.'''
    flat = sum((o - 1) if won else -1 for _, o, won in bets)
    bank, peak, max_dd = bankroll, bankroll, 0.0
    for p, o, won in bets:
        k = max(0.0, (p * (o - 1) - (1 - p)) / (o - 1)) * kelly_fraction
        stake = bank * k
        bank += stake * (o - 1) if won else -stake
        peak = max(peak, bank)
        max_dd = max(max_dd, 1 - bank / peak)
    return {'bets': len(bets), 'wins': sum(w for *_, w in bets), 'flat_profit': flat,
            'flat_roi': flat / len(bets) if bets else 0.0, 'kelly_end': bank, 'kelly_max_dd': max_dd}


def report(conn, run_id, min_edge, kelly_fraction):
    run = conn.execute('SELECT * FROM backtest_runs WHERE id = ?', (run_id,)).fetchone()
    if run is None:
        raise SystemExit(f"[Error] No backtest run {run_id}.")
    rows = load_run(conn, run_id)
    n = len(rows)
    acc = {k: 0.0 for k in ['m_ll', 'm_rps', 'm_brier', 'm_hit', 'k_ll', 'k_rps', 'k_brier', 'k_hit',
                            'm_ou_ll', 'm_ou_hit', 'k_ou_ll', 'k_ou_hit']}
    with_odds = 0
    bets = {('1x2', 'bet365'): [], ('1x2', 'best'): [], ('ou', 'bet365'): [], ('ou', 'best'): []}

    for r in rows:
        hg, ag = r['home_goals'], r['away_goals']
        outcome = 0 if hg > ag else 1 if hg == ag else 2
        over = hg + ag > 2
        raw = [r['prob_h'], r['prob_d'], r['prob_a']]
        p = [v / sum(raw) for v in raw]
        po = r['prob_over_25'] / (r['prob_over_25'] + r['prob_under_25'])
        ll, rps, brier = scores_1x2(p, outcome)
        acc['m_ll'] += ll; acc['m_rps'] += rps; acc['m_brier'] += brier
        acc['m_hit'] += p.index(max(p)) == outcome
        acc['m_ou_ll'] += -math.log(po if over else 1 - po)
        acc['m_ou_hit'] += (po > 0.5) == over

        if r['b_h'] is None or r['b_o'] is None:
            continue
        with_odds += 1
        q = no_margin([r['b_h'], r['b_d'], r['b_a']])
        qo = no_margin([r['b_o'], r['b_u']])[0]
        ll, rps, brier = scores_1x2(q, outcome)
        acc['k_ll'] += ll; acc['k_rps'] += rps; acc['k_brier'] += brier
        acc['k_hit'] += q.index(max(q)) == outcome
        acc['k_ou_ll'] += -math.log(qo if over else 1 - qo)
        acc['k_ou_hit'] += (qo > 0.5) == over

        # One bet per market per match: the outcome with the highest edge, if above min_edge
        for src, cols in (('bet365', 'b'), ('best', 'x')):
            o1 = [r[f'{cols}_h'], r[f'{cols}_d'], r[f'{cols}_a']]
            evs = [p[k] * o1[k] - 1 for k in range(3)]
            k = evs.index(max(evs))
            if evs[k] > min_edge:
                bets[('1x2', src)].append((p[k], o1[k], k == outcome))
            ou = [(po, r[f'{cols}_o'], over), (1 - po, r[f'{cols}_u'], not over)]
            best = max(ou, key=lambda t: t[0] * t[1])
            if best[0] * best[1] - 1 > min_edge:
                bets[('ou', src)].append(best)

    m = {k: v / n for k, v in acc.items() if k.startswith('m_')}
    k = {key: v / with_odds for key, v in acc.items() if key.startswith('k_')} if with_odds else {}
    print(f"\n{'='*78}\nRun {run_id}: season {run['season']}, trained on {run['train_seasons']}, "
          f"rho {run['rho']}, seed {run['seed']} ({run['created_at']})\n{'='*78}")
    print(f"{n} matches predicted; {with_odds} have Bet365 odds.\n")
    print(f"{'':28}{'Model':>10}{'Bet365':>10}{'Guessing':>10}")
    print(f"{'Home/draw/away log loss':28}{m['m_ll']:>10.4f}{k.get('k_ll', float('nan')):>10.4f}{math.log(3):>10.4f}")
    print(f"{'Home/draw/away RPS':28}{m['m_rps']:>10.4f}{k.get('k_rps', float('nan')):>10.4f}")
    print(f"{'Home/draw/away Brier':28}{m['m_brier']:>10.4f}{k.get('k_brier', float('nan')):>10.4f}{2/3:>10.4f}")
    print(f"{'Favourite won':28}{m['m_hit']:>10.1%}{k.get('k_hit', float('nan')):>10.1%}")
    print(f"{'Over/under 2.5 log loss':28}{m['m_ou_ll']:>10.4f}{k.get('k_ou_ll', float('nan')):>10.4f}{math.log(2):>10.4f}")
    print(f"{'Over/under 2.5 right':28}{m['m_ou_hit']:>10.1%}{k.get('k_ou_hit', float('nan')):>10.1%}{0.5:>10.1%}")
    if with_odds and with_odds < n:
        print(f"(Bet365 column covers only the {with_odds} matches with odds; model column covers all {n}.)")

    print(f"\nBetting whenever the model's edge is above {min_edge:.0%} (one bet per market per match):")
    print(f"{'Market':16}{'Odds':8}{'Bets':>6}{'Won':>6}{'Flat profit':>13}{'ROI':>8}"
          f"{f'{kelly_fraction:g} Kelly 1000':>16}{'Max drop':>10}")
    for (market, src), bs in bets.items():
        s = simulate(bs, kelly_fraction)
        name = 'Home/draw/away' if market == '1x2' else 'Over/under 2.5'
        print(f"{name:16}{src:8}{s['bets']:>6}{s['wins']:>6}{s['flat_profit']:>+13.1f}{s['flat_roi']:>+8.1%}"
              f"{s['kelly_end']:>16.0f}{s['kelly_max_dd']:>10.0%}")
    print("Flat profit is in units of 1 stake. 'best' is the highest price across ~17 bookmakers, "
          "which no single account gets every time.")


def export_csv(conn, run_id):
    '''Writes the CSVs that backtest/viz_1x2.py and backtest/viz_ou.py read.'''
    import csv
    rows = load_run(conn, run_id)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    p1, p2 = (os.path.join(out_dir, f'backtest_run{run_id}_{kind}.csv') for kind in ('1x2', 'ou2_5'))
    names = ['H', 'D', 'A']
    with open(p1, 'w', newline='') as f1, open(p2, 'w', newline='') as f2:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        w1.writerow(['Gameweek', 'Home Team', 'Away Team', 'Score', 'True Result', 'Pred Result',
                     'Home Win %', 'Draw %', 'Away Win %', 'Log Loss', 'Brier Score', 'RPS'])
        w2.writerow(['Gameweek', 'Match', 'Score', 'Total Goals', 'True O/U', 'Pred O/U',
                     'Over %', 'Under %', 'Log Loss', 'Brier Score'])
        for r in rows:
            hg, ag = r['home_goals'], r['away_goals']
            outcome = 0 if hg > ag else 1 if hg == ag else 2
            raw = [r['prob_h'], r['prob_d'], r['prob_a']]
            p = [v / sum(raw) for v in raw]
            ll, rps, brier = scores_1x2(p, outcome)
            w1.writerow([r['slot'], r['home'], r['away'], f'{hg}-{ag}', names[outcome], names[p.index(max(p))],
                         f'{p[0]*100:.1f}%', f'{p[1]*100:.1f}%', f'{p[2]*100:.1f}%',
                         round(ll, 4), round(brier, 4), round(rps, 4)])
            po = r['prob_over_25'] / (r['prob_over_25'] + r['prob_under_25'])
            over = hg + ag > 2
            w2.writerow([r['slot'], f"{r['home']} vs {r['away']}", f'{hg}-{ag}', hg + ag,
                         'Over' if over else 'Under', 'Over' if po >= 0.5 else 'Under',
                         f'{po*100:.1f}%', f'{(1-po)*100:.1f}%',
                         round(-math.log(po if over else 1 - po), 4), round((po - over) ** 2, 4)])
    print(f"Exported {p1}\n         {p2}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Walk-forward backtest against Bet365 odds")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--season', nargs='+', help="Season(s) to replay, e.g. 2425 2526")
    group.add_argument('--report', type=int, help="Re-report a saved run by its id")
    group.add_argument('--list', action='store_true', help="List saved runs")
    parser.add_argument('--train', nargs='+',
                        help="Season(s) to fit the preseason prior on (default: the season before each test season)")
    parser.add_argument('--rho', type=float, default=RHO, help=f"Dixon-Coles rho (default {RHO})")
    parser.add_argument('--seed', type=int, default=42, help="Random seed, so a run can be repeated (default 42)")
    parser.add_argument('--shot-weight', type=float, default=db.SHOT_WEIGHT,
                        help=f"Blend shots on target into the rating evidence, 0-1 (default {db.SHOT_WEIGHT}; 0 = goals only)")
    parser.add_argument('--note', help="Free-text note saved with the run")
    parser.add_argument('--min-edge', type=float, default=0.0, help="Bet only above this expected value (default 0 = any edge)")
    parser.add_argument('--kelly', type=float, default=0.25, help="Kelly fraction for the bankroll simulation (default 0.25)")
    parser.add_argument('--export', action='store_true', help="Also write CSVs for viz_1x2.py and viz_ou.py")
    args = parser.parse_args()

    conn = db.connect()
    db.SHOT_WEIGHT = args.shot_weight
    if args.season:
        args.note = f"{args.note + '; ' if args.note else ''}shot_weight={args.shot_weight}"
    if args.list:
        for r in conn.execute('''SELECT r.*, COUNT(p.match_id) AS n FROM backtest_runs r
                                 LEFT JOIN backtest_predictions p ON p.run_id = r.id GROUP BY r.id ORDER BY r.id'''):
            print(f"run {r['id']:>3}  season {r['season']}  train {r['train_seasons']:<10} rho {r['rho']:<8} "
                  f"seed {r['seed']}  {r['n']} matches  {r['created_at']}  {r['note'] or ''}")
        raise SystemExit
    run_ids = [args.report] if args.report else []
    for season in args.season or []:
        prev = f"{int(season[:2]) - 1:02d}{int(season[2:]) - 1:02d}"
        run_ids.append(run_backtest(conn, season, args.train or [prev], args.rho, args.seed, args.note))
    for run_id in run_ids:
        report(conn, run_id, args.min_edge, args.kelly)
        if args.export:
            export_csv(conn, run_id)
