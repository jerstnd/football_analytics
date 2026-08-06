import os 
import json
import csv
import math
import random
import statistics

'''
Data fetching
'''
with open('data/team_mapping.json', 'r') as f:
    TEAMS = {int(k): v for k, v in json.load(f).items()}

NUM_TEAMS = len(TEAMS)

PREVIOUS_SEASON_MATCHES = []
with open('data/pl2325.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # skip header row

    for row in reader:
        match_tuple = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        PREVIOUS_SEASON_MATCHES.append(match_tuple)  


def get_prior(rating):
    return -0.5 * (rating ** 2) / 10000.0

def get_mixture_prior(rating, tier_mean, tier_sigma):
    # A massive variance (1 * 10^4) creates a "flat" bell curve.
    # This prevents the prior from influencing the cold-start data.
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
    # FIX : Anchor the model to the Premier League average of ~1.4 goals per team
    # math.log(1.4) = 0.3364
    INTERCEPT = 0.3364
    
    t_h = math.exp(INTERCEPT + h_adv + att[h_id] + defn[a_id])
    t_a = math.exp(INTERCEPT + att[a_id] + defn[h_id])

    return t_h, t_a

def run_pmcmc(matches, iterations=15000, step_size=0.1):
    print(f"Running MCMC for {iterations} iterations...")
    
    c_home = 0.0
    c_att = [0.0] * NUM_TEAMS
    c_def = [0.0] * NUM_TEAMS

    team_tiers_att = [1] * NUM_TEAMS
    tier_means_att = [-0.50, 0.0, 0.50]
    tier_sigma_att = [0.20, 0.20, 0.20]
    
    traces = {"home": [], "att": [[] for _ in range(NUM_TEAMS)], "def": [[] for _ in range(NUM_TEAMS)]}
    
    for _ in range(iterations):
        
    # Categorical Sweep (Sort teams into tiers)
        for t in range(NUM_TEAMS):  # <-- ADDED: The missing loop for team 't'
            log_probs = []
                
            # Test the team's current attack rating against all 3 tiers
            for tier in range(3):
                # Ensure you are calling your specialized 3-argument prior function here
                lp = get_mixture_prior(c_att[t], tier_means_att[tier], tier_sigma_att[tier])
                log_probs.append(lp)
                
            # Convert log-probabilities to standard percentages
            max_lp = max(log_probs)
            weights = [math.exp(lp - max_lp) for lp in log_probs]
            sum_weights = sum(weights)
            probs = [w / sum_weights for w in weights]
                
            # Assign the team to a new tier probabilistically
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
        for t in range(NUM_TEAMS):
           # --- ATTACK (Uses the Mixture Model Prior) ---
            p_att = list(c_att); p_att[t] += random.gauss(0, step_size)
            ll_diff = get_likelihood(matches, c_home, p_att, c_def) - get_likelihood(matches, c_home, c_att, c_def)
            
            # Fetch the specific tier assigned to this team from Sweep 1
            current_tier = team_tiers_att[t]
            t_mean = tier_means_att[current_tier]
            t_sigma = tier_sigma_att[current_tier]
            
            # Evaluate against their assigned tier, NOT a flat prior
            pr_diff = get_mixture_prior(p_att[t], t_mean, t_sigma) - get_mixture_prior(c_att[t], t_mean, t_sigma)
            
            if ll_diff + pr_diff > 0 or random.random() < math.exp(max(min(ll_diff + pr_diff, 0), -100)):
                c_att = p_att

            # --- DEFENSE (Remains standard unless you build team_tiers_def) ---
            p_def = list(c_def); p_def[t] += random.gauss(0, step_size)
            ll_diff = get_likelihood(matches, c_home, c_att, p_def) - get_likelihood(matches, c_home, c_att, c_def)
            pr_diff = get_prior(p_def[t]) - get_prior(c_def[t]) # Standard flat prior
            
            if ll_diff + pr_diff > 0 or random.random() < math.exp(max(min(ll_diff + pr_diff, 0), -100)):
                c_def = p_def

        # Anchor: Sum-to-Zero
        m_att, m_def = sum(c_att)/NUM_TEAMS, sum(c_def)/NUM_TEAMS
        c_att = [a - m_att for a in c_att]; c_def = [d - m_def for d in c_def]
        
        traces["home"].append(c_home)
        for t in range(NUM_TEAMS):
            traces["att"][t].append(c_att[t]); traces["def"][t].append(c_def[t])
            
    return traces

''' 
EXECUTION & EXPORT
'''

if __name__ == '__main__':
    traces = run_pmcmc(PREVIOUS_SEASON_MATCHES)

    burn_in = int(len(traces['home']) * 0.2)
    state_artifact = {'home_adv': {}, 'teams': {}}

    state_artifact['home_adv']['mean'] = statistics.median(traces['home'][burn_in:])
    state_artifact['home_adv']['var'] = statistics.variance(traces['home'][burn_in:])

    for t in range(NUM_TEAMS):
        state_artifact['teams'][str(t)] = {
            'name' : TEAMS[t],
            'att_mean' : statistics.median(traces['att'][t][burn_in:]),
            'att_var' : statistics.variance(traces['att'][t][burn_in:]),
            'def_mean' : statistics.median(traces['def'][t][burn_in:]),
            'def_var' : statistics.variance(traces['def'][t][burn_in:])
        }

    output_filename = os.path.join('state', 'ratings_backtest.json')
    with open(output_filename, 'w') as f:
        json.dump(state_artifact, f, indent=4)