import json
import math
import random
import statistics
import sys
import os
import csv

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
    num_teams = len(prior_state['teams'])

    # Start the walk exactly where we left off last week
    c_home = prior_state["home_adv"]["mean"]
    c_att = [prior_state["teams"][str(i)]["att_mean"] for i in range(num_teams)]
    c_def = [prior_state["teams"][str(i)]["def_mean"] for i in range(num_teams)]

    traces = {"home": [], "att": [[] for _ in range(num_teams)], "def": [[] for _ in range(num_teams)]}

    for _ in range(iterations):
        # Home Sweep
        p_home = c_home + random.gauss(0, step_size)
        ll_diff = get_likelihood(matches, p_home, c_att, c_def) - get_likelihood(matches, c_home, c_att, c_def)
        pr_diff = get_prior(p_home, prior_state["home_adv"]["mean"], prior_state["home_adv"]["var"]) - \
                  get_prior(c_home, prior_state["home_adv"]["mean"], prior_state["home_adv"]["var"])
        if ll_diff + pr_diff > 0 or random.random() < math.exp(max(min(ll_diff + pr_diff, 0), -100)):
            c_home = p_home

        # Team Sweeps
        for t in range(num_teams):
            p_att = list(c_att); p_att[t] += random.gauss(0, step_size)
            ll_diff = get_likelihood(matches, c_home, p_att, c_def) - get_likelihood(matches, c_home, c_att, c_def)
            pr_diff = get_prior(p_att[t], prior_state["teams"][str(t)]["att_mean"], prior_state["teams"][str(t)]["att_var"]) - \
                      get_prior(c_att[t], prior_state["teams"][str(t)]["att_mean"], prior_state["teams"][str(t)]["att_var"])
            if ll_diff + pr_diff > 0 or random.random() < math.exp(max(min(ll_diff + pr_diff, 0), -100)):
                c_att = p_att

            p_def = list(c_def); p_def[t] += random.gauss(0, step_size)
            ll_diff = get_likelihood(matches, c_home, c_att, p_def) - get_likelihood(matches, c_home, c_att, c_def)
            pr_diff = get_prior(p_def[t], prior_state["teams"][str(t)]["def_mean"], prior_state["teams"][str(t)]["def_var"]) - \
                      get_prior(c_def[t], prior_state["teams"][str(t)]["def_mean"], prior_state["teams"][str(t)]["def_var"])
            if ll_diff + pr_diff > 0 or random.random() < math.exp(max(min(ll_diff + pr_diff, 0), -100)):
                c_def = p_def

        # Anchor
        m_att, m_def = sum(c_att)/num_teams, sum(c_def)/num_teams
        c_att = [a - m_att for a in c_att]; c_def = [d - m_def for d in c_def]

        traces['home'].append(c_home)
        for t in range(num_teams):
            traces['att'][t].append(c_att[t]); traces["def"][t].append(c_def[t])

    return traces

def load_matches(filename):
    matches = []
    with open(filename, mode='r') as f:
        reader = csv.reader(f)
        next(reader, None) # skip header
        for row in reader:
            matches.append((int(row[0]), int(row[1]), int(row[2]), int(row[3])))
    return matches

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python w_bhm.py <new_matches.csv> <previous_ratings.json>')
        sys.exit(1)

    csv_file = sys.argv[1]
    json_file = sys.argv[2]

    print(f"Loading Prior Memory: {json_file}")
    with open(json_file, 'r') as f:
        prior_state = json.load(f)
        
    print(f"Loading New Match Evidence: {csv_file}")
    weekly_matches = load_matches(csv_file)
    
    print("Running Bayesian Update...")
    traces = run_mcmc(weekly_matches, prior_state)

    # 5. Calculate New State & Apply Evolutionary Noise
    burn_in = int(len(traces["home"]) * 0.2)
    time_decay = 0.005 # Injects uncertainty to adapt to sudden form changes
    num_teams = len(prior_state["teams"])

    new_state = {"home_adv": {}, "teams": {}}
    new_state["home_adv"]["mean"] = statistics.median(traces["home"][burn_in:])
    new_state["home_adv"]["var"] = min(statistics.variance(traces["home"][burn_in:]) + time_decay, 0.05)
    
    for t in range(num_teams):
        new_state["teams"][str(t)] = {
            "name": prior_state["teams"][str(t)]["name"],
            "att_mean": statistics.median(traces["att"][t][burn_in:]),
            "att_var": min(statistics.variance(traces["att"][t][burn_in:]) + time_decay, 0.05),
            "def_mean": statistics.median(traces["def"][t][burn_in:]),
            "def_var": min(statistics.variance(traces["def"][t][burn_in:]) + time_decay, 0.05)
        }

    # --- MODIFIED: Use os.path.basename to strip away any existing 'state/' folder prefix ---
    base_name = os.path.basename(json_file).replace(".json", "_updated.json")
    new_filename = os.path.join('state', base_name)
    
    with open(new_filename, 'w') as f:
        json.dump(new_state, f, indent=4)
        
    print(f"Update complete! Hand-off state saved to: {new_filename}")

