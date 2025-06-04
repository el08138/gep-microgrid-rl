"""
Created on Sat Mar 23 20:34:12 2019
Execute all the programs sequentially
@author: Stamatis
"""

b_choices_all = [[300, 1000, 3000]]
versions_to_create, last_version, start_from, end_to, run_all = len(b_choices_all), 0, 0, 1, False
number_data_all, number_runs_all, data_max_all = versions_to_create * [100], versions_to_create * [100], versions_to_create * [1]
number_hospitals_all, number_schools_all, number_homes_all, solar_panels_all, wind_turbines_all = versions_to_create * [2], versions_to_create * [5], versions_to_create * [300], versions_to_create * [6000], versions_to_create * [10]
superposed_scenario_all = versions_to_create * [True]
crit_load_all = [[0.8, 0.6, 0.4] for _ in range(versions_to_create)]
voll_all = [[25, 17, 8] for _ in range(versions_to_create)]
number_episodes_all = versions_to_create * [10 ** 6]
ready_to_go = True

if ready_to_go:
    for i in range(last_version + start_from, last_version + end_to):
        number_data, number_runs, data_max, b_choices, crit_load, voll, number_episodes, version, number_hospitals, number_schools, number_homes, solar_panels, wind_turbines, superposed_scenario = number_data_all[i-last_version], number_runs_all[i-last_version], data_max_all[i-last_version], b_choices_all[i-last_version], crit_load_all[i-last_version], voll_all[i-last_version], number_episodes_all[i-last_version], i + 1, number_hospitals_all[i-last_version], number_schools_all[i-last_version], number_homes_all[i-last_version], solar_panels_all[i-last_version], wind_turbines_all[i-last_version], superposed_scenario_all[i-last_version]
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\first_step_initialization.py").read())
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\import_datasets.py").read())
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\outage_cost_simulation.py").read())
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\regression_model.py").read())
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\rl_algorithm.py").read())
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\quick_test.py").read())
else:
    for i in range(last_version + start_from, last_version + end_to):
        number_data, number_runs, data_max, b_choices, crit_load, voll, number_episodes, version, number_hospitals, number_schools, number_homes, solar_panels, wind_turbines, superposed_scenario = 10, 1, data_max_all[i-last_version], b_choices_all[i-last_version], crit_load_all[i-last_version], voll_all[i-last_version], 100, i + 1, number_hospitals_all[i-last_version], number_schools_all[i-last_version], number_homes_all[i-last_version], solar_panels_all[i-last_version], wind_turbines_all[i-last_version], superposed_scenario_all[i-last_version]
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\first_step_initialization.py").read())
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\import_datasets.py").read())
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\outage_cost_simulation.py").read())
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\regression_model.py").read())
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\rl_algorithm.py").read())
        exec(open(r"C:\Users\stsia\Desktop\Rutgers\Research\Reinforcement Learning\Stam\Journal Paper 1\Code\quick_test.py").read())
