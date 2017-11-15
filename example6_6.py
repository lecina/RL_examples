import numpy as np
import random
import matplotlib.pyplot as plt


"""
	Example 6.6 of Reinforcement Learning: an introduction.

	Summary.
	Implementation of three different on-policy and off-policy algorithms:
	Sarsa, Q-learning and Expected Sarsa. 
	The main differences are that: 1) the states above the cliff have a reward -1 
	when they choose the cliff and the cliff is considered as a state with a single
	action with reward -100 that returns you to the initial position
	and 2) the number of rows can change dynamicaly
	I observe larger deviations compared to the textbook (as also pointed by other
	users)

	Sarsa is an on-policy method that uses the action value of the next state
	to update the current action value. The estimation of the action value
	function, will incorporate the effects of the chosen strategy (e.g. e-greedy).
	Expected Sarsa uses the expected Q value of the next state to 

	Q-learning uses the optimal value (greedy choice) of the next state (e-greedy
	choice) to update the current action value. For this reason, it is an off-policy
	method. This means that the learnt Q will not incorporate the effects of the 
	chosen policy for the exploration.
"""

class States:
	"""
		Class that contains all states and topology
		Each state has its action, and rewards
	"""
	def __init__(self, n, default_actions, do_not_increase_rows=False):
		self.ncols = n
		self.nrows = 4

		self.states = [[] for i in range(self.nrows)]
		for i in range(len(self.states)):
			self.states[i].extend([State(default_actions) for i in range(self.ncols)])

		self.nstates = self.ncols*self.nrows

		self.default_actions = default_actions

		self.do_not_increase_rows = do_not_increase_rows

	def print_optimal_policy(self):
		for i in range(self.nrows)[::-1]:
			print str(i).ljust(4,' '),
			for j in range(self.ncols):
				state = self.states[i][j]
				optimal_action_num = np.argmax(state.get_action_values())
				action = state.get_actions()[optimal_action_num]
				print '{:3s}'.format(action.name),
			print ""


	def add_row(self):
		self.states.append([State(self.default_actions) for i in range(self.ncols)])
		self.nrows += 1
		self.update_nstates()
		
	def update_nstates(self):
		self.nstates = self.ncols * self.nrows

	def add_state(self, row, col, actions):
		while self.nrows <= row:
			self.add_row()

		self.states[row][col] = State(actions)

	def __str__(self):
		s = ""
		for i, row in enumerate(self.states):
			for j, state in enumerate(row):
				s += "State: (%d, %d), actions: %s\n"%(i, j, state.__str__())
		return s

	def get_states(self):
		return self.states

	def get_state(self, row, col):
		while row >= self.nrows:
			self.add_row()
		return self.states[row][col]

	def take_action(self, row, col, action_number):
		"""
			Takes action number "action_number" of state (row,col)
			Note that (row,col) must be in bounds. 
			However, the returning (row,col) may be out of bounds 
			when  "do_not_increase_rows" = false
		"""
		if row >= self.nrows or col >= self.ncols:
			print "Invalid state"
			return

		state = self.states[row][col]

		if len(state.actions) <= action_number:
			print "Invalid action"
			return

		action = state.actions[action_number]

		if action.name == ACTION_TYPE_TO_STRING["left"]: 
			col =  max(col - 1, 0)
		elif action.name == ACTION_TYPE_TO_STRING["right"]: 
			col = min(col + 1, self.ncols-1)
		elif action.name == ACTION_TYPE_TO_STRING["up"]: 
			if self.do_not_increase_rows == True:
				row = min(row + 1, self.nrows-1)
			else:
				row = row + 1 #we assume no upper limit
		elif action.name == ACTION_TYPE_TO_STRING["down"]: 
			row = max(row - 1, 0)
		elif action.name == ACTION_TYPE_TO_STRING["fall_into_cliff"]:
			col = 0
			row = 0
		else:
			return

		return row, col, action.reward

	def game_finished(self, row, col):
		return row == 0 and col == self.ncols - 1

class State:
	"""
		Contains the behavior of each single step: contains list of actions with their rewards and the expected action-value
	"""
	def __init__(self, actions, action_value=[]):
		self.Q = action_value

		self.actions = []
		nactions = len(actions)
		for i in range(nactions):
			action, reward = actions[i]

			self.actions.append(Action(action, reward))

			if action_value == []:
				self.Q = [0 for i in range(nactions)]

	def get_actions(self):
		return self.actions

	def get_action_values(self):
		return self.Q

	def set_action_value(self, action_number, new_value):
		self.Q[action_number] = new_value

	def __str__(self):
		s = ["%s %.2f"%(self.actions[i].__str__(), self.Q[i]) for i in range(len(self.actions))]
		if s == []:
			return "None"
		else:
			return " ".join(s)

class Action:
	"""
		Action description, corresponding reward (only one for the moment) and probability
	"""
	def __init__(self, name="", reward=None):
		self.name = name
		self.reward = reward
	
	def __str__(self):
		return "%s"%(self.name)

ACTION_TYPE_TO_STRING = {
	"left" : "l",
	"right" : "r",
	"up" : "u",
	"down" : "d",
	"fall_into_cliff" : "f",
	"finish" : "F"
}

from abc import ABCMeta, abstractmethod
class Strategy:
	def __init__(self, params):
		self.type = "Parent_class"
		self.params = params

	@abstractmethod
	def get_action(self, state):
		pass

	@abstractmethod
	def get_policy(self, state):
		pass


class Greedy(Strategy):
	def __init__(self, params):
		self.type = "greedy"
		self.params = None

	def get_action(self, state):
		return np.argmax(state.get_action_values())

	def get_policy(self, state):
		policy = [0]*len(state.get_actions())
		policy[self.get_action(state)] = 1
		return policy

class E_greedy(Strategy):
	def __init__(self, params):
		self.type = "epsilon_greedy"
		self.params = params

	def get_action(self, state):
		if random.random() < self.params.get_epsilon():
			return np.random.choice(len(state.get_action_values()))
		else:
			return np.argmax(state.get_action_values())

	def get_policy_for_state(self, state):
		random_prob = self.params.get_epsilon()

		nactions = len(state.get_actions())
		policy = [float(random_prob)/nactions]*nactions

		best_action_prob = 1 - random_prob
		best_action_number = np.argmax(state.get_action_values())
		policy[best_action_number] += best_action_prob

		return policy

class Strategy_params:
	def __init__(self):
		self.epsilon = None

	def set_epsilon(self, epsilon):
		self.epsilon = epsilon

	def get_epsilon(self):
		return self.epsilon

def play_game(initial_state, states):
	"""
		initial_state is a tuple (row, col)
	"""
	finished = False
	rewards = [0]
	visited_states = [initial_state]
	while not finished:
		row, col, r = states.get

		visited_states.append((row, col))
		rewards.append(r)

		if state.game_finished(row,col):
			finished = True

from abc import ABCMeta, abstractmethod
class LearningStrategy:
	def __init__(self):
		self.type = "Parent_class"

	@abstractmethod
	def run_single_episode(self, initial_state, states_object, alpha, gamma):
		pass

class Sarsa(LearningStrategy):
	def __init__(self):
		self.type = "SARSA"

	def run_single_episode(self, initial_state, states, alpha=0.1, gamma = 1):
		"""
			initial_state is a tuple (row, col)
		"""
		finished = False
		rewards = []
		visited_states = [initial_state]

		params = Strategy_params()	
		params.set_epsilon(0.1)

		egreedy = E_greedy(params)

		row = initial_state[0]
		col = initial_state[1]

		state = states.get_state(row, col)
		action_number = egreedy.get_action(state)
		nfalls = 0
		while not finished:
			old_row = row
			old_col = col

			row, col, r = states.take_action(row, col, action_number)
			
			new_state = states.get_state(row, col)
			new_action_number = egreedy.get_action(new_state)
			new_action_value = new_state.get_action_values()[new_action_number]

			prev_action_value = state.get_action_values()[action_number]
			updated_prev_action_value = prev_action_value + alpha * (r + gamma * new_action_value - prev_action_value)
			state.set_action_value(action_number, updated_prev_action_value)

			state = new_state
			action_number = new_action_number
			
			visited_states.append((row, col))
			rewards.append(r)

			if states.game_finished(row,col):
				finished = True

			if row == 0 and col > 0 and col < 11:
				nfalls += 1

		return visited_states, rewards, nfalls

class Qlearning(LearningStrategy):
	def __ini__(self):
		self.type = "Q-Learning" 

	def run_single_episode(self, initial_state, states, alpha=0.1, gamma = 1):
		"""
			initial_state is a tuple (row, col)
		"""
		finished = False
		rewards = []
		visited_states = [initial_state]

		params = Strategy_params()	
		params.set_epsilon(0.1)

		egreedy = E_greedy(params)
		greedy = Greedy(None)

		row = initial_state[0]
		col = initial_state[1]

		nfalls = 0
		while not finished:
			state = states.get_state(row, col)
			action_number = egreedy.get_action(state)

			old_row = row
			old_col = col
			row, col, r = states.take_action(row, col, action_number)

			new_state = states.get_state(row, col)
			new_action_number = greedy.get_action(new_state)
			new_action_value = new_state.get_action_values()[new_action_number]

			prev_action_value = state.get_action_values()[action_number]
			updated_prev_action_value = prev_action_value + alpha * (r + gamma * new_action_value - prev_action_value)
			state.set_action_value(action_number, updated_prev_action_value)

			state = new_state
			
			visited_states.append((row, col))
			rewards.append(r)

			if states.game_finished(row,col):
				finished = True

			if row == 0 and col > 0 and col < 11:
				nfalls += 1

		return visited_states, rewards, nfalls

class Expected_sarsa(LearningStrategy):
	def __ini__(self):
		self.type = "Expected SARSA" 

	def run_single_episode(self, initial_state, states, alpha=0.1, gamma = 1):
		"""
			initial_state is a tuple (row, col)
		"""
		finished = False
		rewards = []
		visited_states = [initial_state]

		params = Strategy_params()	
		params.set_epsilon(0.1)

		egreedy = E_greedy(params)

		row = initial_state[0]
		col = initial_state[1]

		nfalls = 0
		while not finished:
			state = states.get_state(row, col)
			action_number = egreedy.get_action(state)

			old_row = row
			old_col = col
			row, col, r = states.take_action(row, col, action_number)

			#We use the expected action value following the egreedy strategy
			new_state = states.get_state(row, col)
			new_action_values = new_state.get_action_values()
			policy = egreedy.get_policy_for_state(new_state)
			expected_action_value = 0
			for prob, action_value in zip(policy, new_action_values):
				expected_action_value += prob*action_value


			prev_action_value = state.get_action_values()[action_number]
			updated_prev_action_value = prev_action_value + alpha * (r + gamma * expected_action_value - prev_action_value)
			state.set_action_value(action_number, updated_prev_action_value)

			state = new_state
			
			visited_states.append((row, col))
			rewards.append(r)

			if states.game_finished(row,col):
				finished = True

			if row == 0 and col > 0 and col < 11:
				nfalls += 1

		return visited_states, rewards, nfalls


def learn(learning_strategy, episodes, initial_states, states_object, alpha, gamma=1):
	total_length = []
	total_rewards = []
	total_nfalls = []
	for i in range(episodes):	
		initial_state = initial_states[i%len(initial_states)]
		visited_states, rewards, nfalls = learning_strategy.run_single_episode(initial_state, states_object, alpha, gamma)

		total_length.append(len(visited_states))
		total_rewards.append(sum(rewards))
		total_nfalls.append(nfalls)

	return total_length, total_rewards, total_nfalls

def build_states(n):
	states = States( 	n, 
						default_actions = [	(ACTION_TYPE_TO_STRING["left"],-1), 
											(ACTION_TYPE_TO_STRING["right"],-1), 
											(ACTION_TYPE_TO_STRING["up"],-1), 
											(ACTION_TYPE_TO_STRING["down"],-1)], 
						do_not_increase_rows = True)

	for i in range(1,n-1):
		states.add_state(0, i, [(ACTION_TYPE_TO_STRING["fall_into_cliff"],-100)])
	states.add_state(0, n-1, [(ACTION_TYPE_TO_STRING["finish"],0)])

	return states

def average_measures(l):
	avg_l = (len(l)-10)*[0]
	for i in range(len(l)-10):
		avg_l[i] = np.mean(l[i:i+10])
	return avg_l

def main():
	seed = 123
	random.seed(seed)
	np.random.seed(seed)

	n = 12
	sarsa_states = build_states(n)

	initial_states = [(0,0)]
	alpha = 0.25
	episodes = 500
	print "SARSA learning"
	sarsa = Sarsa()
	sarsa_length, sarsa_rewards, sarsa_nfalls = learn(sarsa, episodes, initial_states, sarsa_states, alpha)

	qlearning_states = build_states(n)
	print "Q-learning"
	qlearning = Qlearning()
	qlearning_length, qlearning_rewards, qlearning_nfalls = learn(qlearning, episodes, initial_states, qlearning_states, alpha)

	expected_sarsa_states = build_states(n)
	print "Expected Sarsa"
	expected_sarsa = Expected_sarsa()
	esarsa_length, esarsa_rewards, esarsa_nfalls = learn(expected_sarsa, episodes, initial_states, expected_sarsa_states, alpha)


	avg_sarsa_length = average_measures(sarsa_length)
	avg_sarsa_rewards = average_measures(sarsa_rewards)
	avg_sarsa_nfalls = average_measures(sarsa_nfalls)

	avg_qlearning_length = average_measures(qlearning_length)
	avg_qlearning_rewards = average_measures(qlearning_rewards)
	avg_qlearning_nfalls = average_measures(qlearning_nfalls)

	avg_esarsa_length = average_measures(esarsa_length)
	avg_esarsa_rewards = average_measures(esarsa_rewards)
	avg_esarsa_nfalls = average_measures(esarsa_nfalls)


	print "Sarsa action-value estimated function"
	print sarsa_states
	print "\n===========\n"
	print "Optimal SARSA policy"
	sarsa_states.print_optimal_policy()
	print "\n===========\n===========\n"

	print "Q-learning action-value estimated function"
	print qlearning_states
	print "\n===========\n"
	print "Optimal Q-learning policy"
	qlearning_states.print_optimal_policy()
	print "\n===========\n===========\n"

	print "Expected Sarsa action-value estimated function"
	print expected_sarsa_states
	print "\n===========\n"
	print "Optimal Expected SARSA policy"
	expected_sarsa_states.print_optimal_policy()

	plt.figure(1)
	plt.subplot(221)
	plt.title("Average Rewards")
	plt.plot(range(len(avg_sarsa_rewards)), avg_sarsa_rewards)
	plt.plot(range(len(avg_qlearning_rewards)), avg_qlearning_rewards)
	plt.plot(range(len(avg_esarsa_rewards)), avg_esarsa_rewards)

	plt.subplot(222)
	plt.title("Average Length")
	plt.plot(range(len(avg_sarsa_length)), avg_sarsa_length)
	plt.plot(range(len(avg_qlearning_length)), avg_qlearning_length)
	plt.plot(range(len(avg_esarsa_length)), avg_esarsa_length)

	plt.subplot(223)
	plt.title("Average Number of falls")
	plt.plot(range(len(avg_sarsa_nfalls)), avg_sarsa_nfalls, label='SARSA')
	plt.plot(range(len(avg_qlearning_nfalls)), avg_qlearning_nfalls, label='Q-learning')
	plt.plot(range(len(avg_esarsa_nfalls)), avg_esarsa_nfalls, label='Expected SARSA')
	plt.legend(bbox_to_anchor=(1.5, 0.5), loc=2, borderaxespad=0.)
	plt.show()

if __name__ == "__main__":
	main()
