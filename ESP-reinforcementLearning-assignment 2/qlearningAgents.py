from game import *
from learningAgents import ReinforcementAgent

import random,util,math
          
class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """



  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    self.epsilon = args['epsilon']
    self.alpha = args['alpha']
    self.gamma = args['gamma']
    self.actionFn = args['actionFn']

    "*** YOUR CODE HERE ***"
    self.q_values = util.Counter()
    self.epsilon_decay = 1 #just a value i used to do some tests
    # Initialize Q-values to 0

  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    "*** YOUR CODE HERE ***"
    "Returns Q(state, action)"
    return self.q_values[(state, action)]
 
    
  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    "Returns max_action Q(state, action)"
    legal_actions = self.getLegalActions(state)
    if not legal_actions:
        return 0.0
    return max(self.getQValue(state, action) for action in legal_actions)

    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    "Compute the best action to take in a state"
    legal_actions = self.getLegalActions(state)
    if not legal_actions:
        return None
    q_values = [self.getQValue(state, action) for action in legal_actions]
    max_q_value = max(q_values)
    max_q_value = self.getValue(state)
    # Breaking ties randomly
    #there might be multiple best actions so just to be sure every bath get explored but it isnt neccesary
    best_actions = [action for action, q_value in zip(legal_actions, q_values) if q_value == max_q_value]
    return random.choice(best_actions)

    
  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """  
    "Compute the action to take based on the ε-greedy policy"
    """
    legal_actions = self.getLegalActions(state)
    if not legal_actions:
        return None
    
    if util.flipCoin(self.epsilon):
        return random.choice(legal_actions)
    else:
        return self.getPolicy(state)
    """
    "Compute the action to take based on the ε-greedy policy"
    legal_actions = self.getLegalActions(state)
    if not legal_actions:
        return None
    
    if util.flipCoin(self.epsilon):
        action = random.choice(legal_actions)
    else:
        action = self.getPolicy(state)
    
    # Decay epsilon after each action
    self.epsilon *= self.epsilon_decay
    
    return action
  
  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "Q-value update rule"
    old_q_value = self.getQValue(state, action)
    next_state_value = self.getValue(nextState)
      
    # Q-Learning update formula
    new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * next_state_value)
    self.q_values[(state, action)] = new_q_value
