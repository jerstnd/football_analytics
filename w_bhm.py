import math
import random
import statistics
import sys
import argparse

import db

def get_prior(rating, prior_mean, prior_variance):
    return -((rating - prior_mean) ** 2) / (2 * prior_variance)

def get_likelihood(matches, h_adv, att, defn):
    ll = 0
    for h_id, a_id, h_g, a_g in matches:
        t_h, t_a = get_theta(h_id, a_id, h_adv, att, defn)
        ll += (h_g * math.log(t_h)) - t_h
        ll += (a_g * math.log(t_a)) - t_a
    return ll

def get_theta(h_id, a_id, h_adv, att, defn):
    INTERCEPT = 0.2231
    
    t_h = math.exp(INTERCEPT + h_adv + att[h_id] + defn[a_id])
    t_a = math.exp(INTERCEPT + att[a_id] + defn[h_id])

    return t_h, t_a

def run_mcmc(matches, prior_state, iterations=10000, step_size=0.1):
    # Team IDs are not contiguous (relegated teams are removed, promoted teams keep
    # their historical IDs), so iterate over the state's own keys.
    team_ids = sorted(int(k) for k in prior_state['teams'])
    priors = {t: prior_state['teams'][str(t)] for t in team_ids}

    # Start the walk exactly where we left off last week
    c_home = prior_state["home_adv"]["mean"]
    c_att = {t: priors[t]["att_mean"] for t in team_ids}
    c_def = {t: priors[t]["def_mean"] for t in team_ids}

    # A team's rating only affects its own matches, so the likelihood ratio for
    # a single-team proposal only needs those matches.
    team_matches = {t: [m for m in matches if t in (m[0], m[1])] for t in team_ids}

    traces = {"home": [], "att": {t: [] for t in team_ids}, "def": {t: [] for t in team_ids}}

    for _ in range(iterations):
        # Home Sweep
        p_home = c_home + random.gauss(0, step_size)
        ll_diff = get_likelihood(matches, p_home, c_att, c_def) - get_likelihood(matches, c_home, c_att, c_def)
        pr_diff = get_prior(p_home, prior_state["home_adv"]["mean"], prior_state["home_adv"]["var"]) -                   get_prior(c_home, prior_state["home_adv"]["mean"], prior_state["home_adv"]["var"])
        if ll_diff + pr_diff > 0 or random.random() < math.exp(max(min(ll_diff + pr_diff, 0), -100)):
            c_home = p_home

        # Team Sweeps
        for t in team_ids:
            local = team_matches[t]

            p_att = dict(c_att); p_att[t] += random.gauss(0, step_size)
            ll_diff = get_likelihood(local, c_home, p_att, c_def) - get_likelihood(local, c_home, c_att, c_def)
            pr_diff = get_prior(p_att[t], priors[t]["att_mean"], priors[t]["att_var"]) -                       get_prior(c_att[t], priors[t]["att_mean"], priors[t]["att_var"])
            if ll_diff + pr_diff > 0 or random.random() < math.exp(max(min(ll_diff + pr_diff, 0), -100)):
                c_att = p_att

            p_def = dict(c_def); p_def[t] += random.gauss(0, step_size)
            ll_diff = get_likelihood(local, c_home, c_att, p_def) - get_likelihood(local, c_home, c_att, c_def)
            pr_diff = get_prior(p_def[t], priors[t]["def_mean"], priors[t]["def_var"]) -                       get_prior(c_def[t], priors[t]["def_mean"], priors[t]["def_var"])
            if ll_diff + pr_diff > 0 or random.random() < math.exp(max(min(ll_diff + pr_diff, 0), -100)):
                c_def = p_def

        # Anchor: Sum-to-Zero over this season's teams
        m_att = sum(c_att.values()) / len(team_ids)
        m_def = sum(c_def.values()) / len(team_ids)
        c_att = {t: a - m_att for t, a in c_att.items()}
        c_def = {t: d - m_def for t, d in c_def.items()}

        traces['home'].append(c_home)
        for t in team_ids:
            traces['att'][t].append(c_att[t]); traces["def"][t].append(c_def[t])

    return traces

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MCMC Weekly State Updater")
    parser.add_argument("--season", required=True, help="Season code, e.g. 2627")
    parser.add_argument("--round", type=int, required=True,
                        help="Round whose results to add; reads ratings of round-1, writes ratings of round")
    args = parser.parse_args()

    conn = db.connect()
    prior_state = db.load_state(conn, args.season, args.round - 1)
    if prior_state is None:
        print(f"[Error] No ratings for season {args.season} round {args.round - 1}. "
              f"Run round {args.round - 1} first (round 0 comes from y_bhm.py).")
        sys.exit(1)

    fixtures = db.get_fixtures(conn, args.season, args.round)
    weekly_matches = db.get_results(conn, args.season, args.round)
    if not weekly_matches:
        print(f"[Error] No results for season {args.season} round {args.round}. Run fetch.py first.")
        sys.exit(1)
    if len(weekly_matches) < len(fixtures):
        print(f"[Warning] Only {len(weekly_matches)} of {len(fixtures)} round {args.round} matches have results; "
              f"the rest will be skipped.")
    late = conn.execute('''SELECT COUNT(*) FROM matches WHERE season = ? AND round < ? AND home_goals IS NULL''',
                        (args.season, args.round)).fetchone()[0]
    if late:
        print(f"[Warning] {late} match(es) from earlier rounds are still unplayed (postponed?). "
              f"Their results won't be included when they are played.")

    unknown = sorted({i for m in weekly_matches for i in m[:2]} - {int(k) for k in prior_state['teams']})
    if unknown:
        print(f"[Error] Team IDs {unknown} are not in the round {args.round - 1} ratings. "
              f"Re-run y_bhm.py to initialise promoted teams.")
        sys.exit(1)

    print(f"Running Bayesian Update on {len(weekly_matches)} matches (season {args.season} round {args.round})...")
    traces = run_mcmc(weekly_matches, prior_state)

    # Calculate New State & Apply Evolutionary Noise
    burn_in = int(len(traces["home"]) * 0.2)
    time_decay = 0.005 # Injects uncertainty to adapt to sudden form changes

    new_state = {"home_adv": {}, "teams": {}}
    new_state["home_adv"]["mean"] = statistics.median(traces["home"][burn_in:])
    new_state["home_adv"]["var"] = min(statistics.variance(traces["home"][burn_in:]) + time_decay, 0.05)

    for t in traces["att"]:
        new_state["teams"][str(t)] = {
            "name": prior_state["teams"][str(t)]["name"],
            "att_mean": statistics.median(traces["att"][t][burn_in:]),
            "att_var": min(statistics.variance(traces["att"][t][burn_in:]) + time_decay, 0.05),
            "def_mean": statistics.median(traces["def"][t][burn_in:]),
            "def_var": min(statistics.variance(traces["def"][t][burn_in:]) + time_decay, 0.05)
        }

    db.save_state(conn, args.season, args.round, new_state)
    print(f"Update complete! Saved ratings: season {args.season}, round {args.round}")
