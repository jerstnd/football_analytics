'''
SQLite storage for the whole pipeline: teams, matches (results + fixtures + odds),
weekly rating states and predictions.

Seasons are 4-character strings ('2526' = 2025/26). Ratings are keyed by
(season, round) where round = number of rounds of evidence included, so round 0
is the preseason prior and round N is the state after round N's results.

The model scripts keep working with the same in-memory state dict they always
used: {'home_adv': {'mean', 'var'}, 'teams': {'<id>': {'name', 'att_mean', ...}}}.
'''
import os
import sqlite3

DB_PATH = os.environ.get(
    'FOOTBALL_DB',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'football.db'))

SCHEMA = '''
CREATE TABLE IF NOT EXISTS teams (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

-- Alternative spellings used by data feeds ("Man Utd", "Spurs", ...)
CREATE TABLE IF NOT EXISTS team_aliases (
    alias   TEXT PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(id)
);

-- One row per fixture. Goals are NULL until the match is played.
-- round is NULL for historical seasons imported without gameweek info.
CREATE TABLE IF NOT EXISTS matches (
    id            INTEGER PRIMARY KEY,
    season        TEXT    NOT NULL,
    round         INTEGER,
    date          TEXT,
    home_id       INTEGER NOT NULL REFERENCES teams(id),
    away_id       INTEGER NOT NULL REFERENCES teams(id),
    home_goals    INTEGER,
    away_goals    INTEGER,
    odds_over_25  REAL,
    odds_under_25 REAL,
    UNIQUE (season, home_id, away_id)
);
CREATE INDEX IF NOT EXISTS idx_matches_round ON matches (season, round);

CREATE TABLE IF NOT EXISTS rating_runs (
    season        TEXT    NOT NULL,
    round         INTEGER NOT NULL,
    home_adv_mean REAL    NOT NULL,
    home_adv_var  REAL    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (season, round)
);

CREATE TABLE IF NOT EXISTS ratings (
    season   TEXT    NOT NULL,
    round    INTEGER NOT NULL,
    team_id  INTEGER NOT NULL REFERENCES teams(id),
    att_mean REAL    NOT NULL,
    att_var  REAL    NOT NULL,
    def_mean REAL    NOT NULL,
    def_var  REAL    NOT NULL,
    PRIMARY KEY (season, round, team_id),
    FOREIGN KEY (season, round) REFERENCES rating_runs (season, round) ON DELETE CASCADE
);

-- Latest prediction per match, and which rating round it was made from.
CREATE TABLE IF NOT EXISTS predictions (
    match_id      INTEGER PRIMARY KEY REFERENCES matches(id),
    rating_round  INTEGER NOT NULL,
    xg_h          REAL NOT NULL,
    xg_a          REAL NOT NULL,
    prob_h        REAL NOT NULL,
    prob_d        REAL NOT NULL,
    prob_a        REAL NOT NULL,
    prob_over_25  REAL NOT NULL,
    prob_under_25 REAL NOT NULL,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Bookmaker odds by source, e.g. 'bet365' (pre-match) and 'best' (highest price
-- across ~17 bookmakers). matches.odds_over_25/odds_under_25 stay the odds you
-- enter yourself for betting.
CREATE TABLE IF NOT EXISTS odds (
    match_id INTEGER NOT NULL REFERENCES matches(id),
    source   TEXT    NOT NULL,
    home     REAL,
    draw     REAL,
    away     REAL,
    over_25  REAL,
    under_25 REAL,
    PRIMARY KEY (match_id, source)
);

-- Walk-forward backtests (backtest/backtest.py). Each run keeps its own
-- predictions so runs with different settings can be compared.
CREATE TABLE IF NOT EXISTS backtest_runs (
    id            INTEGER PRIMARY KEY,
    season        TEXT    NOT NULL,
    train_seasons TEXT    NOT NULL,
    rho           REAL    NOT NULL,
    seed          INTEGER,
    note          TEXT,
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backtest_predictions (
    run_id        INTEGER NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    match_id      INTEGER NOT NULL REFERENCES matches(id),
    slot          INTEGER NOT NULL,   -- update step the match was predicted in (by date, see backtest.py)
    xg_h          REAL NOT NULL,
    xg_a          REAL NOT NULL,
    prob_h        REAL NOT NULL,
    prob_d        REAL NOT NULL,
    prob_a        REAL NOT NULL,
    prob_over_25  REAL NOT NULL,
    prob_under_25 REAL NOT NULL,
    PRIMARY KEY (run_id, match_id)
);

-- Kelly-sized orders produced by kelly_sizing.py.
CREATE TABLE IF NOT EXISTS orders (
    match_id   INTEGER PRIMARY KEY REFERENCES matches(id),
    action     TEXT NOT NULL,
    prob       REAL NOT NULL,
    odds       REAL NOT NULL,
    ev         REAL NOT NULL,
    adj_kelly  REAL NOT NULL,
    stake      REAL NOT NULL,
    bankroll   REAL NOT NULL,
    fraction   REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
'''


def connect(path=None):
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    conn.executescript(SCHEMA)
    return conn


# ---------- teams ----------

def team_names(conn):
    return {r['id']: r['name'] for r in conn.execute('SELECT id, name FROM teams')}


def resolve_team(conn, name):
    '''Returns the team id for a canonical name or alias, or None.'''
    name = name.strip()
    row = conn.execute('SELECT id FROM teams WHERE name = ?', (name,)).fetchone()
    if row is None:
        row = conn.execute('SELECT team_id AS id FROM team_aliases WHERE alias = ?', (name,)).fetchone()
    return row['id'] if row else None


def add_team(conn, name):
    cur = conn.execute('INSERT INTO teams (name) VALUES (?)', (name.strip(),))
    return cur.lastrowid


# ---------- matches ----------

def get_results(conn, season, round=None):
    '''Played matches as (home_id, away_id, home_goals, away_goals) tuples.'''
    sql = ('SELECT home_id, away_id, home_goals, away_goals FROM matches '
           'WHERE season = ? AND home_goals IS NOT NULL')
    params = [season]
    if round is not None:
        sql += ' AND round = ?'
        params.append(round)
    return [tuple(r) for r in conn.execute(sql + ' ORDER BY date, id', params)]


def get_fixtures(conn, season, round):
    '''All matches of a round (played or not), with team names and odds.'''
    return conn.execute('''
        SELECT m.id, m.home_id, m.away_id, h.name AS home, a.name AS away,
               m.home_goals, m.away_goals, m.odds_over_25, m.odds_under_25
        FROM matches m
        JOIN teams h ON h.id = m.home_id
        JOIN teams a ON a.id = m.away_id
        WHERE m.season = ? AND m.round = ?
        ORDER BY m.date, m.id''', (season, round)).fetchall()


def season_team_ids(conn, season):
    rows = conn.execute('SELECT home_id FROM matches WHERE season = ? '
                        'UNION SELECT away_id FROM matches WHERE season = ?', (season, season))
    return sorted(r[0] for r in rows)


# ---------- ratings ----------

def save_state(conn, season, round, state):
    with conn:
        conn.execute('DELETE FROM rating_runs WHERE season = ? AND round = ?', (season, round))
        conn.execute('INSERT INTO rating_runs (season, round, home_adv_mean, home_adv_var) VALUES (?, ?, ?, ?)',
                     (season, round, state['home_adv']['mean'], state['home_adv']['var']))
        conn.executemany(
            'INSERT INTO ratings (season, round, team_id, att_mean, att_var, def_mean, def_var) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            [(season, round, int(tid), t['att_mean'], t['att_var'], t['def_mean'], t['def_var'])
             for tid, t in state['teams'].items()])


def load_state(conn, season, round):
    '''Returns the state dict for (season, round), or None if it doesn't exist.'''
    run = conn.execute('SELECT home_adv_mean, home_adv_var FROM rating_runs WHERE season = ? AND round = ?',
                       (season, round)).fetchone()
    if run is None:
        return None
    rows = conn.execute('''
        SELECT r.team_id, t.name, r.att_mean, r.att_var, r.def_mean, r.def_var
        FROM ratings r JOIN teams t ON t.id = r.team_id
        WHERE r.season = ? AND r.round = ? ORDER BY r.team_id''', (season, round))
    return {
        'home_adv': {'mean': run['home_adv_mean'], 'var': run['home_adv_var']},
        'teams': {str(r['team_id']): {'name': r['name'], 'att_mean': r['att_mean'], 'att_var': r['att_var'],
                                      'def_mean': r['def_mean'], 'def_var': r['def_var']} for r in rows}
    }


def latest_round(conn, season):
    row = conn.execute('SELECT MAX(round) FROM rating_runs WHERE season = ?', (season,)).fetchone()
    return row[0]


# ---------- predictions ----------

def save_predictions(conn, rows):
    '''rows: dicts with match_id, rating_round, xg_h, xg_a, prob_h, prob_d, prob_a,
    prob_over_25, prob_under_25. Replaces any earlier prediction for the same match.'''
    with conn:
        conn.executemany('''
            INSERT OR REPLACE INTO predictions
                (match_id, rating_round, xg_h, xg_a, prob_h, prob_d, prob_a, prob_over_25, prob_under_25)
            VALUES (:match_id, :rating_round, :xg_h, :xg_a, :prob_h, :prob_d, :prob_a,
                    :prob_over_25, :prob_under_25)''', rows)


def get_predictions(conn, season, round):
    return conn.execute('''
        SELECT p.*, h.name AS home, a.name AS away, m.odds_over_25, m.odds_under_25
        FROM predictions p
        JOIN matches m ON m.id = p.match_id
        JOIN teams h ON h.id = m.home_id
        JOIN teams a ON a.id = m.away_id
        WHERE m.season = ? AND m.round = ?
        ORDER BY m.date, m.id''', (season, round)).fetchall()
