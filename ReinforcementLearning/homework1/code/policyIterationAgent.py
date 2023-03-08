import util
from abstractAgent import Agent

class PolicyIterationAgent(Agent):
    """An agent that takes a Markov decision process on initialization
    and runs policy iteration for a given number of iterations.

    Hint: Test your code with commands like `python main.py -a policy -i 100 -k 10`.
    """

    def __init__(self, mdp, discount = 0.9, epsilon=0.001, iterations = 100):
        self.mdp = mdp
        self.discount = discount
        self.epsilon = epsilon  # For examing the convergence of policy iteration
        self.iterations = iterations # The policy iteration will run AT MOST these steps
        self.values = util.Counter() # You need to keep the record of all state values here
        self.policy = dict()
        self.runPolicyIteration()

    def runPolicyIteration(self):
        """ YOUR CODE HERE """

        ## initialization of the policy
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            self.policy[state] = self.mdp.getPossibleActions(state)[0]

        for i in range(self.iterations):
            max_delta = 0
            not_change = False

            ## Policy Evaluation
            while True:
                new_values = util.Counter()
                for state in self.mdp.getStates():
                    if self.mdp.isTerminal(state):
                        continue
                    new_values[state] = self.getQValue(state, self.policy[state])
                    delta = new_values[state] - self.values[state]
                    if delta > max_delta:
                        max_delta = delta
                self.values = new_values
                if max_delta < self.epsilon:
                    break
                max_delta = 0
            
            ## Policy Improvement
            new_policy = {state : self.computeActionFromValues(state)
                        for state in self.mdp.getStates()
                        }
            
            if new_policy == self.policy:
                not_change = True
            self.policy = new_policy
        
            if not_change:
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
        return self.policy[state]

    def getAction(self, state):
        return self.policy[state]

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)