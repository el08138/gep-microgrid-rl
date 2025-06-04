"""
Created on Sat Feb 23 19:15:13 2019
Dynamic storage investment in microgrid using reinforcement learning
@author: Stamatis
"""

""" seed for reproducibility """
np.random.seed(initial_seed)
random.seed(initial_seed)

""" load table or create new """
load_or_create_table = 'load'
train_q, train_double_q = True, False
if need_to_update: version = 22

""" load table """
if load_or_create_table == 'load':

    if need_to_update: number_episodes = 10 ** 6 # number of episodes    
    if train_q:
        with open('v' + str(version) + '_q_table_e' + str(number_episodes) + '.pkl', 'rb') as f:
            q_table = pickle.load(f)
        with open('v' + str(version) + '_learning_curve_q_e' + str(number_episodes) + '.pkl', 'rb') as f:
            learning_curve_q = pickle.load(f)
    if train_double_q:
        with open('v' + str(version) + '_q1_table_e' + str(number_episodes) + '.pkl', 'rb') as f:
            q1_table = pickle.load(f)
        with open('v' + str(version) + '_q2_table_e' + str(number_episodes) + '.pkl', 'rb') as f:
            q2_table = pickle.load(f)
        with open('v' + str(version) + '_learning_curve_double_q_e' + str(number_episodes) + '.pkl', 'rb') as f:
            learning_curve_double_q = pickle.load(f)
        
""" create table """
if load_or_create_table == 'create':

    """ hyperparameters """
    if need_to_update: number_episodes = 1 * 10 ** 6 # number of episodes
    batch_num_for_fig = 100
    gamma = 0.9 # discount_factor
    min_alpha, min_epsilon, gamma = 0.02, 0.02, 1 # min threshold for learning rate, min threshold for exploration / exploitation trade-off, discount rate and initialization of progress bar
    alphas, epsilons = np.linspace(1, min_alpha, number_episodes), np.linspace(1, min_epsilon, number_episodes) # learning rate decay and exploration / exploitation trade-off decay
    
    """ financial parameters """
    interest_rate, payments_year = 0.02, 12
    li_yearly_pmt, la_yearly_pmt, vr_yearly_pmt, fw_yearly_pmt = [-payments_year * np.pmt(interest_rate / payments_year, li_life[i] * payments_year, li_price[i]) for i in range(decision_periods)], [-payments_year * np.pmt(interest_rate / payments_year, la_life[j] * payments_year, la_price[j]) for j in range(decision_periods)], [-payments_year * np.pmt(interest_rate / payments_year, vr_life[k] * payments_year, vr_price[k]) for k in range(decision_periods)], [-payments_year * np.pmt(interest_rate / payments_year, fw_life[l] * payments_year, fw_price[l]) for l in range(decision_periods)]
    tech_yearly_pmt = [li_yearly_pmt, la_yearly_pmt, vr_yearly_pmt, fw_yearly_pmt]

    """ states definition """
    q_table, q1_table, q2_table = dict(), dict(), dict()
    start_state = [0]
    for i, x in enumerate(tech_price):
        start_state.append(x[0])
    for i in range(tech):
        start_state.append(0)
    start_state = tuple(start_state)
    total_number_states = np.sum([((a + 1) ** tech) * ((a * b_levels + 1) ** tech) for a in range(decision_periods)])
    # example: state = [0, li_price, la_price, vr_price, fw_price, li_cap, la_cap, vr_cap, fw_cap]
    
    """ actions definiton """
    actions = list(np.linspace(0, b_levels * tech, b_levels * tech + 1))
    actions = list(map(int, actions))
    len_actions = len(actions)
    total_number_states_actions = total_number_states * len_actions
    # actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # action_tuples = [0, (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
    # example: action = 5, action_tuple = (1,2)
    
    """ state transition """
    def state_transition(state, action):    
        """ next state computation """
        next_state = list(state)
        next_state[0] += 1 # periods go up by 1
        if action != 0:
            action_tuple = (int((action - 1) / 3), (action % 3) * (action % 3 != 0) + 3 * (action % 3 == 0))
            next_state[tech+1+action_tuple[0]] += b_choices[action_tuple[1]-1] # in the case I decide to invest in a storage type
        for i in range(tech): # price transitions for every battery technology
            if random.uniform(0, 1) < transition_prob[i][tech_price[i].index(next_state[i+1])]: next_state[i+1] = tech_price[i][tech_price[i].index(next_state[i+1])+1]
        return tuple(next_state)
    
    """ reward computation """
    def reward_computation(state, action):
        x_input = [state[0]]
        for i in range(tech):
            x_input.append(state[tech+1+i])
        if action == 0: 
            investment_cost = 0
        else: 
            action_tuple = (int((action - 1) / 3), (action % 3) * (action % 3 != 0) + 3 * (action % 3 == 0))
            investment_cost = (decision_periods - state[0]) * years_in_period * b_choices[action_tuple[1]-1] * tech_yearly_pmt[action_tuple[0]][tech_price[action_tuple[0]].index(state[action_tuple[0]+1])]
            x_input[action_tuple[0]+1] += b_choices[action_tuple[1]-1]
        try:
            outage_cost = cost_regressor.predict(poly.transform([x_input]))[0]
        except:
            outage_cost = cost_regressor.predict([x_input])[0]
        reward = - investment_cost - outage_cost
        return reward
    
    """ q value function """
    def q(state, action=None):
        if state not in q_table:
            q_table[state] = dict()
        if action is None:
            return q_table[state]
        if action not in q_table[state]:
            q_table[state][action] = 0
        return q_table[state][action]
    
    """ q1 value function """
    def q1(state, action=None):
        if state not in q1_table:
            q1_table[state] = dict()
        if state not in q2_table:
            q2_table[state] = dict()
        if action is None:
            return q1_table[state]
        if action not in q1_table[state]:
            q1_table[state][action] = 0
        if action not in q2_table[state]:
            q2_table[state][action] = 0
        return q1_table[state][action]
    
    """ q2 value function """
    def q2(state, action=None):
        if state not in q1_table:
            q1_table[state] = dict()
        if state not in q2_table:
            q2_table[state] = dict()
        if action is None:
            return q2_table[state]
        if action not in q1_table[state]:
            q1_table[state][action] = 0
        if action not in q2_table[state]:
            q2_table[state][action] = 0
        return q2_table[state][action]
    
    """ q choosing actions """
    def q_choose_action(state, epsilon):
        if random.uniform(0, 1) < epsilon or not bool(q(state)):
            return random.choice(actions) 
        else:
            return max(q(state).keys(), key=(lambda k: q(state)[k]))          
    
    """ double q choosing actions """
    def double_q_choose_action(state, epsilon):
        if random.uniform(0, 1) < epsilon or not bool(q1(state)) or not bool(q2(state)):
            return random.choice(actions) 
        else:
            return max({x: q1(state).get(x, 0) + q2(state).get(x, 0) for x in set(q1(state)).union(q2(state))}.keys(), key=(lambda k: {x: q1(state).get(x, 0) + q2(state).get(x, 0) for x in set(q1(state)).union(q2(state))}[k])) 
        
    """ training q-learning algorithm """
    if train_q:
        saved_rewards, saved_batch_rewards = list(), list()
        fig = plt.figure()
        for episode in tqdm(range(number_episodes)):
            state = start_state
            total_reward = 0
            alpha, epsilon = alphas[episode], epsilons[episode]
            for _ in range(decision_periods):
                action = q_choose_action(state, epsilon)
                next_state, reward = state_transition(state, action), reward_computation(state, action)
                total_reward += reward
                try:
                    max_next_state = q(next_state)[max(q(next_state).keys(), key=(lambda k: q(next_state)[k]))]
                except:
                    max_next_state = 0
                q(state)[action] = q(state, action) + alpha * (reward + gamma *  max_next_state - q(state, action))
                state = next_state
            print(f'Episode {episode + 1}: Total Cost --> {-total_reward}')
            saved_rewards.append(total_reward)
            if (episode + 1) % (number_episodes / batch_num_for_fig) == 0:
                saved_batch_rewards.append(np.mean(saved_rewards))
                saved_rewards = list()
        try:
            learning_curve_q = [list(np.linspace(1, len(saved_batch_rewards), len(saved_batch_rewards))), saved_batch_rewards]
            plt.plot(learning_curve_q[0], learning_curve_q[1])
            plt.show()
            with open('v' + str(version) + '_learning_curve_q_e' + str(number_episodes) + '.pkl', 'wb') as f:
                pickle.dump(learning_curve_q, f)
        except:
            pass
        with open('v' + str(version) + '_q_table_e' + str(number_episodes) + '.pkl', 'wb') as f:
            pickle.dump(q_table, f)
        
    """ training double q-learning algorithm """
    if train_double_q:
        saved_rewards = list()
        fig = plt.figure()
        for episode in tqdm(range(number_episodes)):
            state = start_state
            total_reward = 0
            alpha, epsilon = alphas[episode], epsilons[episode]
            for _ in range(decision_periods):
                action = double_q_choose_action(state, epsilon)
                next_state, reward = state_transition(state, action), reward_computation(state, action)
                total_reward += reward
                if random.uniform(0, 1) < 0.5:
                    try:
                        max_next_state = q2(next_state)[max(q2(next_state).keys(), key=(lambda k: q2(next_state)[k]))]
                    except:
                        max_next_state = 0
                    q1(state)[action] = q1(state, action) + alpha * (reward + gamma *  max_next_state - q1(state, action))
                else:
                    try:
                        max_next_state = q1(next_state)[max(q1(next_state).keys(), key=(lambda k: q1(next_state)[k]))]
                    except:
                        max_next_state = 0
                    q2(state)[action] = q2(state, action) + alpha * (reward + gamma *  max_next_state - q2(state, action))
                state = next_state
            print(f'Episode {episode + 1}: Total Cost --> {-total_reward}')
            saved_rewards.append(total_reward)
            if (episode + 1) % (number_episodes / batch_num_for_fig) == 0:
                saved_batch_rewards.append(np.mean(saved_rewards))
                saved_rewards = list()
        try:
            learning_curve_double_q = [list(np.linspace(1, len(saved_batch_rewards), len(saved_batch_rewards))), saved_batch_rewards]
            plt.plot(learning_curve_double_q[0], learning_curve_double_q[1])
            plt.show()
            with open('v' + str(version) + '_learning_curve_double_q_e' + str(number_episodes) + '.pkl', 'wb') as f:
                pickle.dump(learning_curve_double_q, f)
        except:
            pass
        with open('v' + str(version) + '_q1_table_e' + str(number_episodes) + '.pkl', 'wb') as f:
            pickle.dump(q1_table, f)
        with open('v' + str(version) + '_q2_table_e' + str(number_episodes) + '.pkl', 'wb') as f:
            pickle.dump(q2_table, f)