import util
from abstractAgent import Agent


class ValueIterationAgent(Agent):
    """An agent that takes a Markov decision process on initialization
    and runs value iteration for a given number of iterations.

    Hint: Test your code with commands like `python main.py -a value -i 100 -k 10`.
    """
    def __init__(self, mdp, discount = 0.9, epsilon=0.001, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.epsilon = epsilon  # For examing the convergence of value iteration
        self.iterations = iterations # The value iteration will run AT MOST these steps
        self.values = util.Counter() # You need to keep the record of all state values here
        self.runValueIteration()

    def runValueIteration(self):
        """ YOUR CODE HERE """

        for i in range(self.iterations):
            max_delta = 0
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                actions = self.mdp.getPossibleActions(state)
                value_list = [self.getQValue(state, action) for action in actions]
                delta = max(value_list) - self.values[state]
                self.values[state] = max(value_list)
                if delta > max_delta:
                    max_delta = delta
                
            
            ## Determine convergence
            if max_delta < self.epsilon:
                break
                
                


    def getValue(self, state):
        """Return the value of the state (computed in __init__)."""
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """Compute the Q-value of action in state from the value function stored in self.values."""

        value = None

        """ YOUR CODE HERE """
        value = 0
        successors = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in successors:
            reward = self.mdp.getReward(state, action, nextState)
            value += prob * (reward + self.discount * self.values[nextState])
        return value

    def computeActionFromValues(self, state):
        """The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """

        bestaction = None

        """ YOUR CODE HERE """
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        Q_values = util.Counter()
        for action in actions:
            Q_values[action] = self.getQValue(state, action)
        bestaction = Q_values.argMax()

        return bestaction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
