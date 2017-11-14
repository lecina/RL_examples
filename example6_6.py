import numpy as np
import random

"""
	Example 6.6 of Reinforcement Learning: an introduction.
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
				s += "State: row:%d, col:%d, actions: %s\n"%(i, j, state.__str__())
		return s

	def get_states(self):
		return self.states

	def get_state(self, row, col):
		while row >= self.nrows:
			self.add_row()
		return self.states[row][col]

	def take_action(self, row, col, action_number):
		"""
			Takes action number "action_number" when in state (row,col)
			Note that (row,col) must be in bounds. However, returning
			(row,col) may be out of bounds if "do_not_increase_rows" = false)
		"""
		if row >= self.nrows or col >= self.ncols:
			print "Invalid state"
			return

		state = self.states[row][col]

		if len(state.actions) <= action_number:
			print "Invalid action"
			return

		action = state.actions[action_number]

		if action.name == "left": 
			col =  max(col - 1, 0)
		elif action.name == "right": 
			col = min(col + 1, self.ncols-1)
		elif action.name == "up": 
			if self.do_not_increase_rows == True:
				row = min(row + 1, self.nrows-1)
			else:
				row = row + 1 #we assume no upper limit
		elif action.name == "down": 
			row = max(row - 1, 0)
		elif action.name == "fall_into_cliff":
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
		return "%s, %.2f"%(self.name, self.reward)

from abc import ABCMeta, abstractmethod
class Strategy:
	def __init__(self, params):
		self.type = "Parent_class"
		self.params = params

	@abstractmethod
	def get_action(self, state):
		pass

class Greedy(Strategy):
	def __init__(self, params):
		self.type = "greedy"
		self.params = None

	def get_action(self, state):
		return np.argmax(state.get_action_values())

class E_greedy(Strategy):
	def __init__(self, params):
		self.type = "epsilon_greedy"
		self.params = params

	def get_action(self, state):
		if random.random() < self.params.get_epsilon():
			return np.random.choice(len(state.get_action_values()))
		else:
			return np.argmax(state.get_action_values())

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

def sarsa(episodes, initial_states, states_object, alpha):
	total_length = []
	total_rewards = []
	for i in range(episodes):	
		initial_state = initial_states[i%len(initial_states)]
		visited_states, rewards = sarsa_single_episode(initial_state, states_object, alpha)

		total_length.append(len(visited_states))
		total_rewards.append(sum(rewards))

		print "Length:", len(visited_states), "rewards:", sum(rewards)

	return total_length, total_rewards

def sarsa_single_episode(initial_state, states, alpha=0.1, gamma = 1):
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
	while not finished:
		old_row = row
		old_col = col

		row, col, r = states.take_action(row, col, action_number)
		
		new_state = states.get_state(row, col)
		new_action_number = egreedy.get_action(new_state)
		new_action_value = new_state.get_action_values()[new_action_number]

		prev_action_value = state.get_action_values()[action_number]
		updated_prev_action_value = prev_action_value + alpha * (r + gamma * new_action_value - prev_action_value)
		#print "(%d,%d) action:%d, Q(S,A)=%.2f (Q(S,A)=%.2f) --> (%d, %d) action:%d, Q(S',A')=%.2f"%(old_row, old_col, action_number, updated_prev_action_value, prev_action_value, row, col, new_action_number, new_action_value)
		state.set_action_value(action_number, updated_prev_action_value)

		state = new_state
		action_number = new_action_number
		
		visited_states.append((row, col))
		rewards.append(r)

		if states.game_finished(row,col):
			finished = True

	return visited_states, rewards

def main():
	random.seed(123)

	n = 12
	states = States(n, default_actions = [("left",-1), ("right",-1), ("up",-1), ("down",-1)], do_not_increase_rows = True)
	for i in range(1,n-1):
		states.add_state(0, i, [("fall_into_cliff",-100)])
	states.add_state(0, n-1, [("finish",0)])
	print states

	initial_states = [(0,0)]
	alpha = 1.0
	episodes = 500
	a,b = sarsa(episodes, initial_states, states, alpha)

	print states

if __name__ == "__main__":
	main()
