import mdp, util
import random
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
  """
  def __init__(self, mdp, discount=0.9, iterations=100):
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter()  # Initialize values for all states as zero

    for _ in range(self.iterations):
        new_values = util.Counter()  # Temporary storage for updated values

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            # List to hold values of all actions for the current state
            all_action_values = []

            for action in self.mdp.getPossibleActions(state):
                action_value = 0  # Expected value for the current action

                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    reward = self.mdp.getReward(state, action, next_state)
                    action_value += prob * (reward + self.discount * self.values[next_state])

                all_action_values.append(action_value)

            # Update the state's value with the maximum action value
            new_values[state] = max(all_action_values)

        self.values = new_values  # Replace old values with the newly computed ones

    

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  """
  def getQValue(self, state, action):
    """"""
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
  "*** YOUR CODE HERE ***"
  
  def getQValue(self, state, action):

    total_q_value = 0

    # Fetch all possible next states and their transition probabilities for the given state and action.
    transitions = self.mdp.getTransitionStatesAndProbs(state, action)

    # Iterate over each possible next state and its probability.
    for next_state, prob in transitions:

        # Calculate the immediate reward for moving to the next state after taking the action.
        immediate_reward = self.mdp.getReward(state, action, next_state)

        # Fetch the value of the next state. This value represents the expected reward from that state onwards.
        future_value = self.values[next_state]

        # Calculate the expected reward for this particular transition by adding the immediate reward 
        # and the discounted future value of the next state.
        expected_reward = immediate_reward + self.discount * future_value

        # Multiply the expected reward by the transition probability.
        q_value_contribution = prob * expected_reward

        # Add this to our total Q-value.
        total_q_value += q_value_contribution

    # Return the total computed Q-value.
    return total_q_value




  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    possible_actions = self.mdp.getPossibleActions(state)
    if not possible_actions:
        return None

    return max(possible_actions, key=lambda action: self.getQValue(state, action))
    """ best_action = None
        highest_q_value = float('-inf')

        for action in possible_actions:
            current_q_value = self.getQValue(state, action)
            if current_q_value > highest_q_value:
              highest_q_value = current_q_value
              best_action = action

        return best_action"""



  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
