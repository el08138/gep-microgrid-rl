"""
Created on Wed Feb 27 14:22:34 2019
Simulation for creating the dataset of outage costs
@author: Stamatis
"""

""" seed for reproducibility """
np.random.seed(initial_seed)
random.seed(initial_seed)

""" load results or create new """
load_or_create_results = 'create'
if need_to_update: version = 22

""" load results """
if load_or_create_results == 'load':
    with open('v' + str(version) + '_sim_results' +  '.json', 'r') as f:
        X, cost = json.load(f)

""" create results """
if load_or_create_results == 'create':
    
    """ outage parameters """
    if superposed_scenario: 
        saifi_extreme, saifi_normal, caidi_extreme, caidi_normal = 1 / 6, 0.84 / 6 + 0.89 / 6 + 0.76 / 6 + 1 / 6 + 1.34 / 6 + 1.1 / 6, 22.55, (1.65 + 1.42 + 1.95 + 1.46 + 1.7) / 5
    else: 
        saifi_extreme, saifi_normal, caidi_extreme, caidi_normal = (1.84 + 0.89 + 0.76 + 1 + 1.34 + 1.1) / (6 * 2), (1.84 + 0.89 + 0.76 + 1 + 1.34 + 1.1) / (6 * 2), (22.55 + 1.65 + 1.42 + 1.95 + 1.46 + 1.7) / 6, (22.55 + 1.65 + 1.42 + 1.95 + 1.46 + 1.7) / 6
    prob_extreme = saifi_extreme / (saifi_extreme + saifi_normal) # probability that an outage belongs to the extreme poisson process
    prob_normal = 1 - prob_extreme
    saifi_total = saifi_extreme + saifi_normal
    scale_total_years = 1 / saifi_total
    scale_total = scale_total_years * 8760 # scale parameter of the superposed poisson process converted in hours
    
    """ reliability parameters """
    lolp_constraint = [0.2, 0.4, 0.6]
    ccp_constraint = [0.8, 0.6, 0.4]
    big_m = 0
    if need_to_update: voll = [25, 17, 8]
    
    """ model parameters """
    if need_to_update: number_data, number_runs, data_max = 100, 100, 1 # only these can be changed, number_data should be divisible by decision_periods
    number_facilities = len(voll) # running time ~ 0.73 sec for each run of every datapoint
    X, cost, lolp, ccp = number_data * [(tech + 1) * [0]], number_data * [0], number_data * [0], number_data * [0]
    
    """ outage generation """
    outage_time, outage_duration = list(), list()
    for i in range(number_runs):
        temp_time, temp_duration, time_sum = list(), list(), 0
        time_sum = round(np.random.exponential(scale_total))
        if time_sum > years_in_period * 8760: 
            print()
            print('The run number ' + str(i) + ' has 0 outages!')
            print()
        while time_sum <= years_in_period * 8760:
            temp_time.append(time_sum)
            if np.random.uniform(0, 1) < prob_extreme: 
                temp_duration.append(1 + np.random.poisson(caidi_extreme - 1))
            else:
                temp_duration.append(1 + np.random.poisson(caidi_normal - 1))
            time_sum += round(np.random.exponential(scale_total))
        outage_time.append(temp_time)
        outage_duration.append(temp_duration)
       
    """ data creation """
    for data_point in tqdm(range(number_data)):
        period = np.random.randint(0, decision_periods)
        random_range = np.random.randint(1, b_levels + 1)
        li_counter, la_counter, vr_counter, fw_counter = random.uniform(0, data_max * b_choices[b_levels-random_range] * b_levels * decision_periods), random.uniform(0, data_max * b_choices[b_levels-random_range] * b_levels * decision_periods), random.uniform(0, data_max * b_choices[b_levels-random_range] * b_levels * decision_periods), random.uniform(0, data_max * b_choices[b_levels-random_range] * b_levels * decision_periods)
        tech_max = [li_counter, la_counter, vr_counter, fw_counter]
        tech_dod_temp, tech_eff_temp = [x[period] for x in tech_dod], [y[period] for y in tech_eff]
        tech_min = [a * (1 - b) for a, b in zip(tech_max, tech_dod_temp)]
        tech_sum_discharge, tech_sum_charge = [a * b * c for a, b, c in zip(tech_max, tech_eff_temp, tech_dod_temp)], [d * (1 / e) * f for d, e, f in zip(tech_max, tech_eff_temp, tech_dod_temp)] # compute the eff (or 1 / eff) * max * dod values for every technology
        tech_rate_discharge, tech_rate_charge = [x / sum(tech_sum_discharge) for x in tech_sum_discharge],  [y / sum(tech_sum_charge) for y in tech_sum_charge] # compute the rate at which every technology should contribute to load discharge or charge
        array_cost, array_lolp, array_ccp = number_runs * [0], number_runs * [number_facilities * [0]], number_runs * [number_facilities * [0]] # the final arrays containing results for every simulation run
        X[data_point] = [period, li_counter, la_counter, vr_counter, fw_counter]
        
        """ loop over every simulation run """
        for run in range(number_runs):
            how_many_outages = len(outage_time[run])
            if how_many_outages == 0:
                array_lolp[run], array_ccp[run], array_cost[run] = number_facilities * [0], number_facilities * [1], 0
                continue
            run_cost, run_lolp, run_ccp = 0, number_facilities * [0], number_facilities * [0] # results for each simulation run
            
            """ loop over every outage in run """
            for timeout, duration in zip(outage_time[run], outage_duration[run]):
                hourly_lolp = number_facilities * [0]
                tech_energy = [initial_soc * x for x in tech_max] # energy stored in each storage technology
                tech_soc = [a / b for a, b in zip(tech_energy, tech_max)] # state of charge for each storage technology
                tech_avail_energy = [(x1 - x2) * x3 for x1, x2, x3 in zip(tech_energy, tech_min, tech_eff_temp)] # available energy in each storage technology
                
                """ loop over every hour in outage """
                for hour in range(duration):
                    if prod[timeout+hour] >= np.sum(dem[period],axis=0)[timeout+hour]: # renewables production enough to satisfy the demand
                        energy_amount = prod[timeout+hour] - np.sum(dem[period],axis=0)[timeout+hour]
                        tech_energy = [min(a + b * energy_amount * c, d) for a, b, c, d in zip(tech_energy, tech_rate_charge, tech_eff_temp, tech_max)]
                        tech_soc = [a / b for a, b in zip(tech_energy, tech_max)]
                        tech_avail_energy = [(x1 - x2) * x3 for x1, x2, x3 in zip(tech_energy, tech_min, tech_eff_temp)]
                    elif prod[timeout+hour] + sum(tech_avail_energy) >= np.sum(dem[period],axis=0)[timeout+hour]: # renewables production together with storage devices enough to satisfy the demand
                        energy_amount = np.sum(dem[period],axis=0)[timeout+hour] - prod[timeout+hour]
                        tech_energy = [a - (b * energy_amount) / c for a, b, c in zip(tech_energy, tech_rate_discharge, tech_eff_temp)]
                        tech_soc = [a / b for a, b in zip(tech_energy, tech_max)]
                        tech_avail_energy = [(x1 - x2) * x3 for x1, x2, x3 in zip(tech_energy, tech_min, tech_eff_temp)]
                    else: # microgrid not able to satisfy the demand
                        jcount, temp_prod = -1, prod[timeout+hour]
                        while prod[timeout+hour] + sum(tech_avail_energy) < np.sum(dem[period][:jcount],axis=0)[timeout+hour]:
                            jcount -= 1
                            if jcount == -number_facilities: break
                        facilities_satisfied = number_facilities + jcount
                        for i in range(number_facilities):
                            if i < facilities_satisfied:
                                if temp_prod >= dem[period][i][timeout+hour]:
                                    temp_prod -= dem[period][i][timeout+hour]
                                else:
                                    energy_amount = dem[period][i][timeout+hour] - temp_prod
                                    temp_prod = 0
                                    tech_energy = [a - (b * energy_amount) / c for a, b, c in zip(tech_energy, tech_rate_discharge, tech_eff_temp)]
                                    tech_soc = [a / b for a, b in zip(tech_energy, tech_max)]
                                    tech_avail_energy = [(x1 - x2) * x3 for x1, x2, x3 in zip(tech_energy, tech_min, tech_eff_temp)]
                            else:
                                hourly_lolp[i] += 1
                                run_cost += dem[period][i][timeout+hour] * voll[i]  
                        energy_amount = temp_prod
                        tech_energy = [min(a + b * energy_amount * c, d) for a, b, c, d in zip(tech_energy, tech_rate_charge, tech_eff_temp, tech_max)]
                        tech_soc = [a / b for a, b in zip(tech_energy, tech_max)]
                        tech_avail_energy = [(x1 - x2) * x3 for x1, x2, x3 in zip(tech_energy, tech_min, tech_eff_temp)]
                        """ end of loop over every hour in outage """
                        
                temp_lolp = [k / duration for k in hourly_lolp]
                temp_ccp = [1 if x <= lolp_constraint[i] else 0 for i, x in enumerate(temp_lolp)]
                run_lolp = [a + b for a, b in zip(run_lolp, temp_lolp)]
                run_ccp = [a + b for a, b in zip(run_ccp, temp_ccp)]
                """ end of loop over every outage in run """
                
            array_lolp[run] = [a / how_many_outages for a in run_lolp]
            array_ccp[run] = [b / how_many_outages for b in run_ccp]
            array_cost[run] = run_cost
            """ end of loop over every simulation run """
        
        lolp[data_point] = np.mean(array_lolp, axis = 0) # final lolp statistic for each system configuration
        ccp[data_point] = np.mean(array_ccp, axis = 0) # final ccp statistic for each system configuration
        cost[data_point] = np.mean(array_cost) + sum([big_m if x < ccp_constraint[i] else 0 for i, x in enumerate(ccp[data_point])]) # final cost statistic for each system configuration
    with open('v' + str(version) + '_sim_results' +  '.json', 'w') as f:
        json.dump([X, cost], f)
