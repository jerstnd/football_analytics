'''
Load match data into football.db.

  python fetch.py --season 2627                       # fixtures + results from fixturedownload.com
  python fetch.py --season 2627 --odds my_odds.csv    # O/U 2.5 odds: HomeTeam,AwayTeam,Odds_Over_25,Odds_Under_25
  python fetch.py --season 2526 --history season-2526.csv   # a football-data.co.uk season CSV (no rounds)
  python fetch.py --season 2526 --market-odds Matches.csv    # Bet365 + best-price odds (Club-Football-Match-Data)
  python fetch.py --season 2627 --shots               # shots on target for played matches, from ESPN

Every mode is an upsert keyed on (season, home team, away team), so re-running is safe.
'''
import sys
import csv
import json
import argparse
import urllib.request

import db

FEED_URL = 'https://fixturedownload.com/feed/json/epl-{year}'
# shortcut: unofficial, keyless ESPN endpoint; matches football-data.co.uk HST/AST on 751/760 2025/26 counts.
# If it breaks, fall back to a hand-typed CSV.
ESPN_URL = 'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={date}'


def team_id_or_add(conn, name, allow_new):
    tid = db.resolve_team(conn, name)
    if tid is None:
        if not allow_new:
            print(f"[Error] Unknown team '{name}'. Add it to teams (or team_aliases if it's another spelling).")
            sys.exit(1)
        tid = db.add_team(conn, name)
        print(f"  [+] New team '{name}' -> ID {tid}  (if this is another spelling of an existing team, "
              f"delete it and add a team_aliases row instead)")
    return tid


def upsert_match(conn, season, home_id, away_id, round=None, date=None, home_goals=None, away_goals=None):
    '''Returns 'new', 'result', 'changed' or 'same'. Never clears fields it wasn't given.'''
    row = conn.execute('SELECT id, round, date, home_goals, away_goals FROM matches '
                       'WHERE season = ? AND home_id = ? AND away_id = ?', (season, home_id, away_id)).fetchone()
    if row is None:
        conn.execute('INSERT INTO matches (season, round, date, home_id, away_id, home_goals, away_goals) '
                     'VALUES (?, ?, ?, ?, ?, ?, ?)', (season, round, date, home_id, away_id, home_goals, away_goals))
        return 'new'

    new = {'round': round if round is not None else row['round'],
           'date': date or row['date'],
           'home_goals': home_goals if home_goals is not None else row['home_goals'],
           'away_goals': away_goals if away_goals is not None else row['away_goals']}
    if all(new[k] == row[k] for k in new):
        return 'same'
    # 'changed' = something we relied on was different (a corrected score, a moved round);
    # filling in a missing date or a rescheduled date is not worth flagging.
    if row['home_goals'] is not None and (new['home_goals'], new['away_goals']) != (row['home_goals'], row['away_goals']):
        status = 'changed'
    elif row['round'] is not None and new['round'] != row['round']:
        status = 'changed'
    elif row['home_goals'] is None and new['home_goals'] is not None:
        status = 'result'
    else:
        status = 'same'
    conn.execute('UPDATE matches SET round = :round, date = :date, home_goals = :home_goals, '
                 'away_goals = :away_goals WHERE id = :id', {**new, 'id': row['id']})
    return status


def fetch_feed(conn, season):
    url = FEED_URL.format(year=2000 + int(season[:2]))
    print(f"Fetching {url}")
    req = urllib.request.Request(url, headers={'User-Agent': 'football-analytics/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        feed = json.load(resp)

    counts = {'new': 0, 'result': 0, 'changed': 0, 'same': 0}
    with conn:
        # New teams are only expected before a season starts (promoted sides).
        allow_new = conn.execute('SELECT COUNT(*) FROM matches WHERE season = ?', (season,)).fetchone()[0] == 0
        for m in feed:
            h = team_id_or_add(conn, m['HomeTeam'], allow_new)
            a = team_id_or_add(conn, m['AwayTeam'], allow_new)
            status = upsert_match(conn, season, h, a, round=int(m['RoundNumber']), date=m['DateUtc'][:10],
                                  home_goals=m['HomeTeamScore'], away_goals=m['AwayTeamScore'])
            counts[status] += 1
            if status == 'changed':
                print(f"  [!] Updated existing match: {m['HomeTeam']} vs {m['AwayTeam']} (round {m['RoundNumber']})")

    played = conn.execute('SELECT MAX(round) FROM matches WHERE season = ? AND round IS NOT NULL AND round NOT IN '
                          '(SELECT round FROM matches WHERE season = ? AND home_goals IS NULL AND round IS NOT NULL)',
                          (season, season)).fetchone()[0]
    print(f"{len(feed)} matches: {counts['new']} new, {counts['result']} new results, "
          f"{counts['changed']} changed, {counts['same']} unchanged.")
    print(f"Latest fully played round: {played}")


def import_odds(conn, season, path):
    n = 0
    with conn, open(path, newline='') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            if not row.get('HomeTeam'):
                continue
            h, a = db.resolve_team(conn, row['HomeTeam']), db.resolve_team(conn, row['AwayTeam'])
            cur = conn.execute('UPDATE matches SET odds_over_25 = ?, odds_under_25 = ? '
                               'WHERE season = ? AND home_id = ? AND away_id = ?',
                               (float(row['Odds_Over_25']) or None, float(row['Odds_Under_25']) or None, season, h, a))
            if cur.rowcount == 0:
                print(f"  [!] No {season} match found for {row['HomeTeam']} vs {row['AwayTeam']}")
            n += cur.rowcount
    print(f"Updated odds for {n} matches.")


def import_market_odds(conn, season, path):
    '''
    Bet365 and best-price odds from the Club-Football-Match-Data Matches.csv
    (github.com/xgabora/Club-Football-Match-Data-2000-2025), which republishes
    football-data.co.uk. Only Premier League (E0) rows of the given season are read,
    and a row is stored only if its score matches the result already in the database.
    '''
    first_year = 2000 + int(season[:2])
    start, end = f'{first_year}-07-01', f'{first_year + 1}-06-30'
    num = lambda v: float(v) if v not in ('', None) else None
    counts = {'stored': 0, 'no_match': 0, 'score_differs': 0}
    with conn, open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['Division'] != 'E0' or not (start <= row['MatchDate'] <= end):
                continue
            h, a = db.resolve_team(conn, row['HomeTeam']), db.resolve_team(conn, row['AwayTeam'])
            m = conn.execute('SELECT id, home_goals, away_goals FROM matches WHERE season = ? AND home_id = ? AND away_id = ?',
                             (season, h, a)).fetchone() if h is not None and a is not None else None
            if m is None:
                counts['no_match'] += 1
                print(f"  [!] No {season} match for {row['HomeTeam']} vs {row['AwayTeam']} ({row['MatchDate']})")
                continue
            if m['home_goals'] is not None and (m['home_goals'], m['away_goals']) != (int(float(row['FTHome'])), int(float(row['FTAway']))):
                counts['score_differs'] += 1
                print(f"  [!] Score differs, skipped: {row['HomeTeam']} vs {row['AwayTeam']} "
                      f"(file {row['FTHome']}-{row['FTAway']}, db {m['home_goals']}-{m['away_goals']})")
                continue
            conn.executemany('INSERT OR REPLACE INTO odds (match_id, source, home, draw, away, over_25, under_25) '
                             'VALUES (?, ?, ?, ?, ?, ?, ?)', [
                (m['id'], 'bet365', num(row['OddHome']), num(row['OddDraw']), num(row['OddAway']),
                 num(row['Over25']), num(row['Under25'])),
                (m['id'], 'best', num(row['MaxHome']), num(row['MaxDraw']), num(row['MaxAway']),
                 num(row['MaxOver25']), num(row['MaxUnder25'])),
            ])
            counts['stored'] += 1
    total = conn.execute('SELECT COUNT(*) FROM matches WHERE season = ?', (season,)).fetchone()[0]
    print(f"{season}: odds stored for {counts['stored']} of {total} matches "
          f"({counts['no_match']} unmatched rows, {counts['score_differs']} score mismatches)")


def import_history(conn, season, path):
    '''football-data.co.uk format: Date (YYYY-MM-DD or DD/MM/YYYY), HomeTeam, AwayTeam, FTHG, FTAG.'''
    counts = {'new': 0, 'result': 0, 'changed': 0, 'same': 0}
    with conn, open(path, newline='') as f:
        for row in csv.DictReader(f):
            if not row.get('HomeTeam'):
                continue
            date = row['Date']
            if '/' in date:
                d, mth, y = date.split('/')
                date = f"{'20' + y if len(y) == 2 else y}-{mth.zfill(2)}-{d.zfill(2)}"
            h = team_id_or_add(conn, row['HomeTeam'], True)
            a = team_id_or_add(conn, row['AwayTeam'], True)
            counts[upsert_match(conn, season, h, a, date=date,
                                home_goals=int(row['FTHG']), away_goals=int(row['FTAG']))] += 1
            if row.get('HST'):
                conn.execute('UPDATE matches SET home_sot = ?, away_sot = ? WHERE season = ? AND home_id = ? AND away_id = ?',
                             (int(row['HST']), int(row['AST']), season, h, a))
    print(f"{path}: {counts}")


def import_shots(conn, season):
    '''Shots on target from ESPN for played matches that don't have them yet, one request per match date.'''
    dates = [r[0] for r in conn.execute('SELECT DISTINCT date FROM matches WHERE season = ? AND home_goals IS NOT NULL '
                                        'AND home_sot IS NULL ORDER BY date', (season,))]
    n = 0
    with conn:
        for date in dates:
            req = urllib.request.Request(ESPN_URL.format(date=date.replace('-', '')),
                                         headers={'User-Agent': 'football-analytics/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                events = json.load(resp)['events']
            for e in events:
                c = e['competitions'][0]
                if c['status']['type']['name'] != 'STATUS_FULL_TIME':
                    continue
                side = {t['homeAway']: t for t in c['competitors']}
                ids = [db.resolve_team(conn, side[k]['team']['displayName']) for k in ('home', 'away')]
                sot = [next((s['displayValue'] for s in side[k].get('statistics', []) if s['name'] == 'shotsOnTarget'), None)
                       for k in ('home', 'away')]
                if None in ids or None in sot:
                    print(f"  [!] Skipped {side['home']['team']['displayName']} vs {side['away']['team']['displayName']} "
                          f"on {date}: unknown team name or no shots data")
                    continue
                n += conn.execute('UPDATE matches SET home_sot = ?, away_sot = ? WHERE season = ? AND home_id = ? '
                                  'AND away_id = ? AND home_goals IS NOT NULL', (int(sot[0]), int(sot[1]), season, *ids)).rowcount
    missing = conn.execute('SELECT COUNT(*) FROM matches WHERE season = ? AND home_goals IS NOT NULL AND home_sot IS NULL',
                           (season,)).fetchone()[0]
    print(f"Shots added for {n} matches; {missing} played match(es) still without shots.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load match data into football.db")
    parser.add_argument('--season', required=True, help="Season code, e.g. 2627")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--odds', help="CSV of O/U 2.5 odds to attach to this season's matches")
    group.add_argument('--history', help="football-data.co.uk season CSV to import")
    group.add_argument('--shots', action='store_true', help="Shots on target for played matches, from ESPN")
    group.add_argument('--market-odds', help="Club-Football-Match-Data Matches.csv: Bet365 and best-price odds")
    args = parser.parse_args()

    conn = db.connect()
    if args.odds:
        import_odds(conn, args.season, args.odds)
    elif args.market_odds:
        import_market_odds(conn, args.season, args.market_odds)
    elif args.shots:
        import_shots(conn, args.season)
    elif args.history:
        import_history(conn, args.season, args.history)
    else:
        fetch_feed(conn, args.season)
