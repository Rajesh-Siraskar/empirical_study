# ----------------------------------------------------------------------------------------------------
# Three key features of any RL algorithm:
# 1. Policy: $\pi_\theta$ = Probablities of all actions, given a state. Parameterized by $\theta$.  
# 2. Objective function: $ \max_\limits{\theta} J(\pi_{\theta}) = \mathop{\mathbb{E}}_{\tau \sim \pi_\theta} [R(\tau)]$
# 3. Method: Way to udate the parameters = Policy Gradient

# Policy gradient numerical computation
# - Plain vanilla: $\nabla_\theta J(\pi_\theta)  = \mathbb{E}_{\tau \sim \pi_\theta} \; [ \; \sum_{t=0}^T R_t(\tau) \; \nabla_\theta \ln \pi_\theta(a_t \vert s_t) \;]$
# - With Monte Carlo sampling and approximation: $\nabla_\theta J(\pi_\theta) \approx [ \; \sum_{t=0}^T R_t(\tau) \; \nabla_\theta \ln \pi_\theta(a_t \vert s_t) \;]$

# - With baseline: $\nabla_\theta J(\theta) \approx [ \; \sum_{t=0}^T (R_t(\tau) - b(s_t)) \; \nabla_\theta \ln \pi_\theta(a_t \vert s_t) \;]$
# - Where, baseline does not change per time-step, it is for the entire trajectory
# - One baseline option: $V^\pi$ - leads to Actor-Critic algorithm
# - Simpler option: Average returns over trajectory: $b = \frac{1}{T}\sum_{t=0}^T R_t(\tau) $

# ### Algorithm
# 1. Initialize $\alpha$, $\gamma$ and $\theta$ i.e. weigths of the NN
# 2. for episodes = 0 to MAX_EPISODES:
#     - sample trajectory $\tau$
#     - set $\nabla_\theta J(\pi_{\theta})$ = 0
#     - for t=0 to T:
#         - $R_t(\tau) = \sum_{t'=t}^{T} \gamma^{t'-t} r'_t$
#         - $\nabla_\theta J(\pi_\theta)  = \mathbb{E}_{\tau \sim \pi_\theta} \; [ \; \sum_{t=0}^T R_t(\tau) \; \nabla_\theta \ln \pi_\theta(a_t \vert s_t) \;]$
#     - end for sampled trajectory
#     - $\theta = \theta + \alpha \nabla_\theta J(\pi_\theta) $
# 3. end for all episodes 

# ### Implementation notes:
# 1. Code is inspired by Laura Graesser's _(LG, 2020 book)_ implementation, however with large modifications
# 2. We use the concept of Agent (instead of 'pi' or 'policy_pi') and a separate network class (i.e. how the function approximator is implemented, it could well be a linear regression)
# 3. **Important concept**: "Loss" in the implementation below, is the "objective" $J$. In our algorithm, we want to **maximize** it. PyTorch's optimizer, by default, MINIMIZES it (as it is called "loss"), we therefore add "-" negate it, so as to maximize it. 
# 4. Also in the final plot, notice "loss" is RISING and follows the rewards, which is expected as "loss" is really being maximized.  
# 5. self.pd.probs = prob. distribution of all actions
# 6. Sum of all possible action probabilities = 1.0
# 7. Note that episodes > 300 start showing repeated patterns. Rewards drop and rise in cylces. 300 is ideal and hence suggested in LG (2020)

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical, Normal
import torch.nn as nn
import torch.optim as optim

### Network class - function approximator
# - Simple one layer MLP

class PolicyNetwork(nn.Module):
    
    # Step 1: Define the network architecture
    def __init__(self, lr, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        
        # 1.1. Define network architecture
        layers = [
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        ]
            
        # layers = [
        #     nn.Linear(input_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, output_dim),
        # ]
        
        # 1.3. Assemble the network and this becomes our "model" i.e function approximation (i.e. "model") 
        self.model = nn.Sequential(*layers)
        
        # 1.2. Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    # Step 2: Feed-forward algo. PyTorch handles back-prop for us, but feed-forward we must provide
    def feed_forward(self, state):
        # Probability distribution (pd) parameters
        pdparam = self.model(state)
        return (pdparam)

### Agent class
# - This will also "hold" the policy network, defined above, by class PolicyNetwork
# - Other agent specific activities:
#     - Learn the "policy" using the "policy network" above
#     - Decide what action to take
#     - On-policy type - so discard previous experiences 

class Agent():
    def __init__(self, input_dim, n_actions, alpha, gamma):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        
        self.log_probs = []
        self.rewards = []
        self.pd = None
        # Create the policy network
        self.policy_network = PolicyNetwork(self.alpha, self.input_dim, self.n_actions)
                
        # On-policy, so discard previous experiences. Empty buffer
        self.onpolicy_reset()
        
        # Call training loop
        # self.learn()
        
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def act(self, state):
        ## Continous action - use Normal pd
        ## pd = Normal(loc=pdparams[0], scale=pdparams[1]) # probability distribution
        
        x = torch.from_numpy(state.astype(np.float32)) # Convert to tensor
        pdparam = self.policy_network.feed_forward(x) # forward pass
        pd = Categorical(logits=pdparam) # probability distribution
        ## Note: 
        # 1. self.pd.probs = prob. distribution of all actions
        # 2. Sum of all possible action probabilities = 1.0
        action = pd.sample() # pi(a|s) in action via pd
        log_prob = pd.log_prob(action) # log prob of pi(a|s)
        self.log_probs.append(log_prob) # store for training
        return (action.item())
    
    ## predict is the function used by Stable-Baselines
    ## We simply call act()
    def predict(self, state):
        predicted_action = self.act(state)
        next_state = None
        return predicted_action, next_state
    
    def learn(self):
        # Inner gradient-ascent loop
        T = len(self.rewards) # Length of a trajectory
        returns = np.empty(T, dtype=np.float32)
        future_returns = 0.0

        # Compute returns
        for t in reversed(range(T)):
            future_returns = self.rewards[t] + self.gamma*future_returns
            returns[t] = future_returns
            
        returns = torch.tensor(returns)
        log_probs = torch.stack(self.log_probs)
        
        loss = torch.sum(- log_probs*returns) # Compute gradient term. Negative for maximizing
        self.policy_network.optimizer.zero_grad()
        loss.backward() # backpropogate and compute gradients
        self.policy_network.optimizer.step() # gradient ascent, update the weights
        return (loss)