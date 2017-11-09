import numpy as np

"""
	Example 4.1 of Reinforcement Learning: an introduction.

	It contains an iterative method to compute the state-value function for a policy
	and the first-visit MC prediction
"""

class States:
	"""
		Class that contains all states with their possible action and rewards
	"""
	def __init__(self, n):
		self.states = [[]]*n*n
		self.n = n
		self.nstates = n*n

	def add_state(self, state, actions):
		self.states[state] = State(actions)

	def __str__(self):
		s = ""
		for i, state in enumerate(self.states):
			s += "State: %d, actions: %s\n"%(i, state.__str__())
		return s

	def get_states(self):
		return self.states

	def get_action_resulting_state(self, state_num, action_number=None):
		if state_num >= self.nstates:
			print "Invalid state"
			return

		state = self.states[state_num]
		if action_number is None:
			if state.actions != []:
				action = np.random.choice(state.actions, p=state.policy)
			else:
				return state_num, 0
		elif len(state.actions) <= action_number:
			print "Invalid action"
			return
		else:
			action = state.actions[action_number]


		row = state_num / self.n
		col = state_num % self.n

		if action.name == "left": 
			col =  max(col - 1, 0)
		elif action.name == "right": 
			col = min(col + 1, self.n-1)
		elif action.name == "up": 
			row = max(row - 1, 0)
		elif action.name == "down": 
			row = min(row + 1, self.n-1)
		else:
			return state_num, 0

		return self.n*row + col, action.reward

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

def iterative_policy_evaluation(states, iterations, gamma=1):
	"""
		Implemented as in page 83 of Reinforcement Learning: an introduction
		(with except that the exit condition is with the number of iterations, and not with V's convergence)
	"""
	V = np.zeros(states.nstates)
	for i in range(iterations):
		v = np.copy(V) #It converges faster if we use partially updated V values (i.e. work with V and not with the copy v). The convergence rate is then affected by the order
		for s, state in enumerate(states.get_states()):
			actions = state.get_actions()
			V[s] = 0
			for a, action in enumerate(actions):
				sprime, r = states.get_action_resulting_state(s,a)
				policy_a_given_s = state.policy[a]
				probability =  1 #the number of "end states" and their reward is unique given an action
				V[s] += policy_a_given_s * probability * (r + gamma * v[sprime])

	return V

def play_game(initial_state, states):
	"""
		Returns list of visited states and the corresponding reward after each action
		Note that len(states) = len(rewards) + 1
	"""

	state = initial_state
	finished = False
	rewards = []
	visited_states = [state]
	while not finished:
		state, r = states.get_action_resulting_state(state)
		visited_states.append(state)
		rewards.append(r)

		if state == 0 or state == 15:
			finished = True

	return visited_states, rewards

def estimate_G(states, rewards):
	found_states = set(states)
	size = max(found_states)
	Gs = []
	for state in found_states:
		first_visit = states.index(state)
		G = sum(rewards[first_visit:])
		Gs.append((state, G))
	return Gs
		

def first_visit_monte_carlo_prediction(states, n):
	nstates = n*n
	estimatedV = np.zeros(nstates)
	nappearances = np.zeros(nstates)
	ngames = 1000
	for i in range(ngames):
		starting_state = i%nstates
		visited_states, rewards = play_game(starting_state, states)
		Gs = estimate_G(visited_states, rewards)

		for state, G in Gs:
			nappearances[state] += 1
			estimatedV[state] += G

	estimatedV /= nappearances

	return estimatedV

def main():
	n = 4 #nstates = n*n
	states = States(n)
	states.add_state(0,[])
	states.add_state(15, [])
	for i in range(1,15):
		states.add_state(i, [("left",-1), ("right",-1), ("up",-1), ("down",-1)])
	print states


	V = iterative_policy_evaluation(states, iterations=100, gamma=1)
	print "Iterative policy evaluation:"
	print V

	#We can check that it corresponds to the average reward using the policy
	estimatedV = first_visit_monte_carlo_prediction(states, n)
	print "Estimated V:"
	print estimatedV
	 


if __name__ == "__main__":
	main()
