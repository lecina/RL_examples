import numpy as np

"""
	Example 6.6 of Reinforcement Learning: an introduction.
"""

class States:
	"""
		Class that contains all states and topology
		Each state has its policy, action, and rewards
	"""
	def __init__(self, n, default_actions):
		self.states = [[]]*n
		for i in range(len(self.states)):
			self.states[i] = [State(default_actions)]*n

		self.ncols = n
		self.nrows = n

		self.nstates = n*n

		self.default_actions = default_actions

	def add_row(self):
		self.states.append([State(self.default_actions)]*self.ncols)
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

	def get_action_resulting_state(self, row, col, action_number=None):
		if row >= self.nrows or col >= self.ncols:
			print "Invalid state"
			return

		while row >= self.nrows:
			states.add_row()

		state = self.states[row][col]

		if action_number is None:
			if state.actions != []:
				action = np.random.choice(state.actions, p=state.policy)
			else:
				return row, col, 0
		elif len(state.actions) <= action_number:
			print "Invalid action"
			return
		else:
			action = state.actions[action_number]


		if action.name == "left": 
			col =  max(col - 1, 0)
		elif action.name == "right": 
			col = min(col + 1, self.n-1)
		elif action.name == "up": 
			row = max(row - 1, 0)
		elif action.name == "down": 
			row = min(row + 1, self.n-1)
		elif action.name == "fall_into_cliff":
			col = 0
			row = 0
		else:
			return

		return row, col, action.reward

class State:
	"""
		Contains the behavior of each single step: contains list of actions with their rewards and the policy
	"""
	def __init__(self, actions, policy=[]):
		self.actions = []
		self.policy = policy

		nactions = len(actions)
		for i in range(nactions):
			action, reward = actions[i]

			if policy == []:
				self.policy = [1./nactions]*nactions

			self.actions.append(Action(action, reward))


	def get_actions(self):
		return self.actions

	def __str__(self):
		s = [action.__str__() for action in self.actions]
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

def main():
	n = 12
	states = States(n, default_actions = [("left",-1), ("right",-1), ("up",-1), ("down",-1)])
	for i in range(1,n-1):
		states.add_state(0, i, [("fall_into_cliff",-100)])
	states.add_state(0, n-1, [("finish",0)])
	print states

if __name__ == "__main__":
	main()
