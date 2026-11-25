import math
import random
import statistics
import argparse

import db

INTERCEPT = 0.2231

# Promoted teams have no top-flight data, so their prior is borrowed from the
# relegated teams' ratings. This variance is deliberately wider than a full-season
# posterior (~0.02) because the relegated teams are only a proxy.
PROMOTED_VAR = 0.1


def get_prior(rating):
    # A massive variance (1 * 10^4) creates a "flat" bell curve.
    return -0.5 * (rating ** 2) / 10000.0


def get_mixture_prior(rating, tier_mean, tier_sigma):
    if tier_sigma <= 0:
        return -float('inf')
    return -0.5 * (((rating - tier_mean) / tier_sigma) ** 2) - math.log(tier_sigma)


def get_likelihood(matches, h_adv, att, defn):
    ll = 0
    for h_id, a_id, h_g, a_g in matches:
        t_h, t_a = get_theta(h_id, a_id, h_adv, att, defn)
        ll += (h_g * math.log(t_h)) - t_h
        ll += (a_g * math.log(t_a)) - t_a
    return ll


def get_theta(h_id, a_id, h_adv, att, defn):
    t_h = math.exp(INTERCEPT + h_adv + att[h_id] + defn[a_id])
    t_a = math.exp(INTERCEPT + att[a_id] + defn[h_id])
    return t_h, t_a


def run_pmcmc(matches, team_ids, iterations=15000, step_size=0.1):
    '''
    team_ids must be exactly the teams that appear in `matches`. A team with no
    matches has no likelihood, so its rating would random-walk and pollute the
    sum-to-zero anchor for everyone else.
    '''
    print(f"Running MCMC for {iterations} iterations over {len(team_ids)} teams...")

    c_home = 0.0
    c_att = {t: 0.0 for t in team_ids}
    c_def = {t: 0.0 for t in team_ids}

    team_tiers_att = {t: 1 for t in team_ids}
    tier_means_att = [-0.50, 0.0, 0.50]
    tier_sigma_att = [0.20, 0.20, 0.20]

    # A team's rating only affects its own matches, so the likelihood ratio for
    # a single-team proposal only needs those matches.
    team_matches = {t: [m for m in matches if t in (m[0], m[1])] for t in team_ids}

    traces = {"home": [], "att": {t: [] for t in team_ids}, "def": {t: [] for t in team_ids}}

    for _ in range(iterations):

        # Categorical Sweep (Sort teams into tiers)
        for t in team_ids:
            log_probs = [get_mixture_prior(c_att[t], tier_means_att[k], tier_sigma_att[k]) for k in range(3)]
            max_lp = max(log_probs)
            weights = [math.exp(lp - max_lp) for lp in log_probs]
            sum_weights = sum(weights)
            probs = [w / sum_weights for w in weights]

            rand_val = random.random()
            if rand_val < probs[0]:
                team_tiers_att[t] = 0
            elif rand_val < probs[0] + probs[1]:
                team_tiers_att[t] = 1
            else:
                team_tiers_att[t] = 2

        # Home Sweep
        p_home = c_home + random.gauss(0, step_size)
        ll_diff = get_likelihood(matches, p_home, c_att, c_def) - get_likelihood(matches, c_home, c_att, c_def)
        pr_diff = get_prior(p_home) - get_prior(c_home)
        if ll_diff + pr_diff > 0 or random.random() < math.exp(max(min(ll_diff + pr_diff, 0), -100)):
            c_home = p_home

        # Team Sweeps
        for t in team_ids:
            local = team_matches[t]

            # --- ATTACK (Uses the Mixture Model Prior) ---
            p_att = dict(c_att); p_att[t] += random.gauss(0, step_size)
            ll_diff = get_likelihood(local, c_home, p_att, c_def) - get_likelihood(local, c_home, c_att, c_def)
            t_mean = tier_means_att[team_tiers_att[t]]
            t_sigma = tier_sigma_att[team_tiers_att[t]]
            pr_diff = get_mixture_prior(p_att[t], t_mean, t_sigma) - get_mixture_prior(c_att[t], t_mean, t_sigma)
            if ll_diff + pr_diff > 0 or random.random() < math.exp(max(min(ll_diff + pr_diff, 0), -100)):
                c_att = p_att

            # --- DEFENSE (Standard flat prior) ---
            p_def = dict(c_def); p_def[t] += random.gauss(0, step_size)
            ll_diff = get_likelihood(local, c_home, c_att, p_def) - get_likelihood(local, c_home, c_att, c_def)
            pr_diff = get_prior(p_def[t]) - get_prior(c_def[t])
            if ll_diff + pr_diff > 0 or random.random() < math.exp(max(min(ll_diff + pr_diff, 0), -100)):
                c_def = p_def

        # Anchor: Sum-to-Zero over the teams that actually played
        m_att = sum(c_att.values()) / len(team_ids)
        m_def = sum(c_def.values()) / len(team_ids)
        c_att = {t: a - m_att for t, a in c_att.items()}
        c_def = {t: d - m_def for t, d in c_def.items()}

        traces["home"].append(c_home)
        for t in team_ids:
            traces["att"][t].append(c_att[t]); traces["def"][t].append(c_def[t])

    return traces


def summarise(traces, team_ids, id_to_name, burn_in_frac=0.2):
    burn_in = int(len(traces['home']) * burn_in_frac)
    state = {'home_adv': {}, 'teams': {}}
    state['home_adv']['mean'] = statistics.median(traces['home'][burn_in:])
    state['home_adv']['var'] = statistics.variance(traces['home'][burn_in:])
    for t in team_ids:
        state['teams'][str(t)] = {
            'name': id_to_name[t],
            'att_mean': statistics.median(traces['att'][t][burn_in:]),
            'att_var': statistics.variance(traces['att'][t][burn_in:]),
            'def_mean': statistics.median(traces['def'][t][burn_in:]),
            'def_var': statistics.variance(traces['def'][t][burn_in:])
        }
    return state


def init_new_season(state, season_team_ids, id_to_name):
    '''
    Turns last season's posterior into this season's prior:
      - drops relegated teams (in last season, not in this one)
      - adds promoted teams with the relegated teams' average rating as prior
    '''
    season_team_ids = {str(t) for t in season_team_ids}
    relegated = sorted(t for t in state['teams'] if t not in season_team_ids)
    promoted = sorted(season_team_ids - set(state['teams']), key=int)

    if promoted and not relegated:
        raise ValueError("Promoted teams found but no relegated teams to borrow a prior from.")

    new_teams = {tid: info for tid, info in state['teams'].items() if tid in season_team_ids}

    if promoted:
        att_prior = statistics.mean(state['teams'][t]['att_mean'] for t in relegated)
        def_prior = statistics.mean(state['teams'][t]['def_mean'] for t in relegated)
        for tid in promoted:
            new_teams[tid] = {
                'name': id_to_name[int(tid)],
                'att_mean': att_prior,
                'att_var': PROMOTED_VAR,
                'def_mean': def_prior,
                'def_var': PROMOTED_VAR
            }

    print(f"Relegated (dropped): {', '.join(state['teams'][t]['name'] for t in relegated) or 'none'}")
    print(f"Promoted (added):    {', '.join(id_to_name[int(t)] for t in promoted) or 'none'}")
    if promoted:
        print(f"Promoted prior:      att {att_prior:+.3f}, def {def_prior:+.3f}, var {PROMOTED_VAR}")
    if len(new_teams) != 20:
        print(f"[Warning] New-season state has {len(new_teams)} teams, expected 20.")

    sorted_teams = dict(sorted(new_teams.items(), key=lambda kv: int(kv[0])))
    return {'home_adv': state['home_adv'], 'teams': sorted_teams}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-season MCMC fit on previous season(s), saved as round 0")
    parser.add_argument("--train", nargs='+', required=True, help="Season(s) to fit on, e.g. 2526")
    parser.add_argument("--season", required=True,
                        help="Season the ratings are the prior for, e.g. 2627. Its teams (from fetch.py) "
                             "decide who is dropped/promoted.")
    args = parser.parse_args()

    conn = db.connect()
    id_to_name = db.team_names(conn)

    matches = [m for s in args.train for m in db.get_results(conn, s)]
    if not matches:
        print(f"[Error] No played matches found for season(s) {args.train}.")
        raise SystemExit(1)
    team_ids = sorted({m[0] for m in matches} | {m[1] for m in matches})

    season_ids = db.season_team_ids(conn, args.season)
    if not season_ids:
        print(f"[Error] No {args.season} fixtures in the database. Run fetch.py --season {args.season} first.")
        raise SystemExit(1)

    traces = run_pmcmc(matches, team_ids)
    state = init_new_season(summarise(traces, team_ids, id_to_name), season_ids, id_to_name)

    db.save_state(conn, args.season, 0, state)
    print(f"Saved preseason ratings: season {args.season}, round 0")
