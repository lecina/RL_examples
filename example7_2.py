import numpy as np
import random
import matplotlib.pyplot as plt

"""
	Example 7.2 of Reinforcement Learning: an introduction.

	Summary.
	Random walk where there are two terminal states and we
	get a +1 reward when we terminate in one of them. Otherwise
	the reward is 0.
"""

class States:
	"""
		Class that contains all states and topology
		Each state has its action, and rewards
	"""
	def __init__(self, n, default_actions):
		self.nstates = n

		self.states = [State(default_actions) for i in range(self.nstates)]

		self.default_actions = default_actions


	def print_optimal_policy(self):
		for j in range(self.nstates):
			state = self.states[j]
			optimal_action_num = np.argmax(state.get_action_values())
			action = state.get_actions()[optimal_action_num]
			print '{:3s}'.format(action.name),
		print ""

	def get_optimal_action_values(self):
		value_state_for_optimal_action = []
		for j in range(self.nstates):
			state = self.states[j]
			optimal_action_value = np.max(state.get_action_values())
			value_state_for_optimal_action.append(optimal_action_value)

		return value_state_for_optimal_action

	def update_state(self, state, actions):
		self.states[state] = State(actions)

	def __str__(self):
		s = ""
		for j, state in enumerate(self.states):
			s += "State: %d, actions: %s\n"%(j, state.__str__())
		return s

	def get_states(self):
		return self.states

	def get_state(self, state):
		if self.nstates > state:
			return self.states[state]
		else:
			print "Invalid state in get_state: state: %d with %d total states"%(state, self.nstates)
			return None

	def take_action(self, state_num, action_number):
		"""
			Takes action number "action_number" of state
		"""
		if state_num >= self.nstates:
			print "Invalid state"
			return

		state = self.states[state_num]

		if len(state.actions) <= action_number:
			print "Invalid action"
			return

		action = state.actions[action_number]

		if action.name == ACTION_TYPE_TO_STRING["left"]: 
			new_state_num = max(state_num - 1, 0)
		elif action.name == ACTION_TYPE_TO_STRING["right"]: 
			new_state_num = min(state_num + 1, self.nstates-1)
		elif action.name == ACTION_TYPE_TO_STRING["finish"]:
			new_state_num = state_num
		else:
			return

		return new_state_num, action.reward

	def game_finished(self, state):
		return state == 0 or state == self.nstates - 1

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
		optimal_actions = np.argwhere(state.get_action_values() == np.amax(state.get_action_values())).flatten()
		return np.random.choice(optimal_actions)

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
			optimal_actions = np.argwhere(state.get_action_values() == np.amax(state.get_action_values())).flatten()
			return np.random.choice(optimal_actions)

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
			initial_state is a number
		"""
		finished = False
		rewards = []
		visited_states = [initial_state]

		params = Strategy_params()	
		params.set_epsilon(0.1)

		egreedy = E_greedy(params)

		state_num = initial_state

		state = states.get_state(state_num)
		action_number = egreedy.get_action(state)
		while not finished:
			state_num, r = states.take_action(state_num, action_number)
			
			new_state = states.get_state(state_num)
			new_action_number = egreedy.get_action(new_state)
			new_action_value = new_state.get_action_values()[new_action_number]

			prev_action_value = state.get_action_values()[action_number]
			updated_prev_action_value = prev_action_value + alpha * (r + gamma * new_action_value - prev_action_value)
			state.set_action_value(action_number, updated_prev_action_value)

			state = new_state
			action_number = new_action_number
			
			visited_states.append(state_num)
			rewards.append(r)

			if states.game_finished(state_num):
				finished = True

		return visited_states, rewards

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

		state_num = initial_state

		while not finished:
			state = states.get_state(state_num)
			action_number = egreedy.get_action(state)

			state_num, r = states.take_action(state_num, action_number)

			new_state = states.get_state(state_num)
			new_action_number = greedy.get_action(new_state)
			new_action_value = new_state.get_action_values()[new_action_number]

			prev_action_value = state.get_action_values()[action_number]
			updated_prev_action_value = prev_action_value + alpha * (r + gamma * new_action_value - prev_action_value)
			state.set_action_value(action_number, updated_prev_action_value)

			#state = new_state
			
			visited_states.append(state_num)
			rewards.append(r)

			if states.game_finished(state_num):
				finished = True

		return visited_states, rewards

class Expected_sarsa(LearningStrategy):
	def __ini__(self):
		self.type = "Expected SARSA" 

	def run_single_episode(self, initial_state, states, alpha=0.1, gamma = 1):
		"""
			initial_state is a number
		"""
		finished = False
		rewards = []
		visited_states = [initial_state]

		params = Strategy_params()	
		params.set_epsilon(0.1)

		egreedy = E_greedy(params)

		state_num = initial_state

		while not finished:
			state = states.get_state(state_num)
			action_number = egreedy.get_action(state)

			old_state_num = state_num
			state_num, r = states.take_action(state_num, action_number)

			#We use the expected action value following the egreedy strategy
			new_state = states.get_state(state_num)
			new_action_values = new_state.get_action_values()
			policy = egreedy.get_policy_for_state(new_state)
			expected_action_value = 0
			for prob, action_value in zip(policy, new_action_values):
				expected_action_value += prob*action_value


			prev_action_value = state.get_action_values()[action_number]
			updated_prev_action_value = prev_action_value + alpha * (r + gamma * expected_action_value - prev_action_value)
			state.set_action_value(action_number, updated_prev_action_value)

			state = new_state
			
			visited_states.append(state_num)
			rewards.append(r)

			if states.game_finished(state_num):
				finished = True

		return visited_states, rewards


def learn(learning_strategy, episodes, initial_states, states_object, alpha, gamma=1):
	total_length = []
	total_rewards = []
	total_finishing_in_zero = []
	for i in range(episodes):	
		initial_state = initial_states[i%len(initial_states)]
		visited_states, rewards = learning_strategy.run_single_episode(initial_state, states_object, alpha, gamma)

		total_length.append(len(visited_states))
		total_rewards.append(sum(rewards))
		total_finishing_in_zero.append(visited_states[-1] == 0)

	return total_length, total_rewards, total_finishing_in_zero

def build_states(n):
	states = States( 	n, 
						default_actions = [	(ACTION_TYPE_TO_STRING["left"],0), 
											(ACTION_TYPE_TO_STRING["right"],0)])

	states.update_state(0, [(ACTION_TYPE_TO_STRING["finish"],0)])
	states.update_state(n-1, [(ACTION_TYPE_TO_STRING["finish"],0)])
	states.update_state(n-2, [(ACTION_TYPE_TO_STRING["left"],0), (ACTION_TYPE_TO_STRING["right"],1)])

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

	n = 7
	sarsa_states = build_states(n)

	initial_states = [n/2]
	#initial_states = range(n)
	alpha = 0.1
	episodes = 100000
	print "SARSA learning"
	sarsa = Sarsa()
	sarsa_length, sarsa_rewards, sarsa_zeros = learn(sarsa, episodes, initial_states, sarsa_states, alpha)

	qlearning_states = build_states(n)
	print "Q-learning"
	qlearning = Qlearning()
	qlearning_length, qlearning_rewards, qlearning_zeros = learn(qlearning, episodes, initial_states, qlearning_states, alpha)

	expected_sarsa_states = build_states(n)
	print "Expected Sarsa"
	expected_sarsa = Expected_sarsa()
	esarsa_length, esarsa_rewards, esarsa_zeros = learn(expected_sarsa, episodes, initial_states, expected_sarsa_states, alpha)


	avg_sarsa_length = average_measures(sarsa_length)
	avg_sarsa_rewards = average_measures(sarsa_rewards)
	avg_sarsa_zeros = average_measures(sarsa_zeros)

	avg_qlearning_length = average_measures(qlearning_length)
	avg_qlearning_rewards = average_measures(qlearning_rewards)
	avg_qlearning_zeros = average_measures(qlearning_zeros)

	avg_esarsa_length = average_measures(esarsa_length)
	avg_esarsa_rewards = average_measures(esarsa_rewards)
	avg_esarsa_zeros = average_measures(esarsa_zeros)

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

	plt.figure(2)
	plt.suptitle("Action value for optimal action")
	plt.subplot(221)
	values_sarsa = sarsa_states.get_optimal_action_values()
	values_sarsa = [values_sarsa]
	plt.imshow(values_sarsa, cmap="hot")
	plt.colorbar()
	plt.title("Sarsa")
	plt.subplot(222)
	values_qlearning = qlearning_states.get_optimal_action_values()
	values_qlearning = [values_qlearning]
	plt.imshow(values_qlearning, cmap="hot")
	plt.colorbar()
	plt.title("Q-learning")
	plt.subplot(223)
	values_esarsa = expected_sarsa_states.get_optimal_action_values()
	values_esarsa = [values_esarsa]
	plt.imshow(values_esarsa, cmap="hot")
	plt.colorbar()
	plt.title("Expected Sarsa")

	plt.show()

if __name__ == "__main__":
	main()
