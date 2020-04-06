# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import random

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        possiblestates = self.mdp.getStates()
        self.values = util.Counter()
        #print(self.values)
        for i in range(self.iterations):
            newvalues = self.values.copy()
            for state in possiblestates:
                possibleVals = []
                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                else:
                    possibleActions = self.mdp.getPossibleActions(state)
                    for action in possibleActions:
                        possibleVals.append(self.computeQValueFromValues(state,action))
                    newvalues[state] = max(possibleVals)
            self.values = newvalues




        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        for tup in self.mdp.getTransitionStatesAndProbs(state, action):
            value += tup[1]*(self.mdp.getReward(state, action, tup[0]) + self.discount*self.values[tup[0]])
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state) or len(possibleActions) == 0:
            return None
        else:
            value = float("-inf")
            bestAction = None
            for action in self.mdp.getPossibleActions(state):
                placeholder = self.computeQValueFromValues(state, action)
                if placeholder > value or value == float("-inf"):
                    value = placeholder
                    bestAction = action
            return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class PolicyIterationAgent(ValueEstimationAgent):
    def __init__(self, mdp, discount = 0.9):
        """
          Your policy iteration agent should take an mdp on
          construction and run policy iteration until convergence

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.values = util.Counter() # A Counter is a dict with default 0
        self.policy = util.Counter()
        self.runPolicyIteration()
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        for tup in self.mdp.getTransitionStatesAndProbs(state, action):
            value += tup[1]*(self.mdp.getReward(state, action, tup[0]) + self.discount*self.values[tup[0]])
        return value

    def runPolicyIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        possiblestates = self.mdp.getStates()
        self.values = util.Counter()
        for state in possiblestates:
            possibleActions = self.mdp.getPossibleActions(state)
            p = len(possibleActions)
            if p > 0:
                self.policy[state] = possibleActions[random.randint(0,p-1)]
            else:
                self.policy[state] = None
        # itr = 0  
        while True:
            newValues = self.values.copy()
            for state in possiblestates:
                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                else:
                    action = self.policy[state]
                    possibleval = self.computeQValueFromValues(state,action)
                    newValues[state] = possibleval
            self.values = newValues
            unchanged = True
            newPolicy = self.policy.copy()
            for state in possiblestates:
                oldaction = self.policy[state]      
                if self.mdp.isTerminal(state) or len(possibleActions) == 0:
                    continue
                else:
                    value = float("-inf")
                    bestAction = None
                    possibleActions = self.mdp.getPossibleActions(state)
                    for action in possibleActions:
                        placeholder = self.computeQValueFromValues(state, action)
                        if placeholder > value:
                            value = placeholder
                            bestAction = action
                    newPolicy[state] = bestAction
                    if oldaction != bestAction:
                        unchanged = False
            if(unchanged):
                print("Terminated")
                break
            else:
                self.policy = newPolicy
            # itr += 1
            # if itr > 100:
            #     break
        return self.policy


                
    


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

