#-----------------------------------------------------------------------------------------------------------------
# Milling Tool wear environment class
# Author: Rajesh Siraskar
# Date: 28-Apr-2023
#
# - Simple actions: Replace / do-not-replace
# - Reward for every step of lengthened life
# - Reward for every step where wear < Threshold
# - Penalty (cost) for replacement
# - Termination condition:
#     - Wear > Threshold
#     - End of Wear data-frame
#     - Episode length >1000

# V 2.0: Open AI Gym compliant. Fix threshold normalization bug. Potential reward function errors
# V 2.1 Modify reward function and simpify to basics
# TO-DO - Re-factor MillingTool_V1 on lines of MillingTool_V2
# V 3.0 - Env. version 3.0 -- PHM multi-state environment
# V 3.2 - Env. version 3.0 -- PHM multi-state environment
# V 3.90 - R +1*indx, -1.2*idx and -2.0*idx > Rep error (0.13) reduced but high normal error (0.8). F-1 0.471:
# V 3.91 - R +1*indx, -1.2*idx and -4.0**idx > Error Rep: 0.00 Normal: 0.98 F-1 0.355
# V 3.92 - R +1*indx, -1.2*idx and -4.0*idx > Error Rep: 0.00 Normal: 1.0 F-1 0.33
# V 3.93 - R +1*indx, -1.2*idx and -1.0*idx > Error Rep: 0.00 Normal: 1.0 F-1 0.33
# V 3.94 - R +1*indx, -1.2*idx and -10 > Error Rep: 0.00 Normal: 1.0 F-1 0.33
# V 3.95: repl - how far from threshold. More far - more penalty
# V.4.00: Add reward function elements to init method. R1, R2 and R3. class MillingTool_MS_V2(gym.Env):
#-----------------------------------------------------------------------------------------------------------------
import gym
from gym import spaces
import pandas as pd
import numpy as np


# *************************
# CODE REVIEW
# *************************
## Single variable State V.2.0
class MillingTool_SS_NT(gym.Env):
    """Custom Milling Tool Wear Environment that follows the Open AI gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, wear_threshold, max_operations, add_noise, breakdown_chance, R1=1.0, R2=-1.0, R3=-100.0):
        print(f'** -- Single-variate env. Terminate on (1) tool breakdown (2) data-end (3) milling operations end.  R1: {R1}, R2: {R2}, R3: {R3}. Noise: {add_noise}. Break-down chance: {breakdown_chance} -- **')

        # Machine data frame properties
        self.df = df
        self.df_length = len(self.df.index)
        self.df_index = 0

        # Milling operation and tool parameters
        self.wear_threshold = wear_threshold
        self.max_operations = max_operations
        self.breakdown_chance = breakdown_chance
        self.add_noise = add_noise
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.reward = 0.0

        # Statistics
        self.ep_length = 0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_rewards_history = []
        self.ep_length_history = []
        self.ep_tool_replaced_history = []

        ## Gym interface Obsercation and Action spaces
        # All features are normalized [0, 1]
        self.min_feature = 0.0
        self.max_feature = 1.0

        # Define state and action limits
        self.low_state = np.array([self.min_feature, self.min_feature], dtype=np.float32)
        self.high_state = np.array([self.max_feature, self.max_feature], dtype=np.float32)

        # Observation and action spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    def step(self, action):
        """
        Args: action. Discrete - 1=Replace tool, 0=Continue milling operation
        Returns: next_state, reward, terminated, truncated , info
        """
        # Get current observation from environment
        self.state = self._get_observation(self.df_index)
        time_step = self.state[0]
        tool_wear = self.state[1]

        # Add white noise for robustness
        # if self.add_noise:
        #     tool_wear += np.random.rand()/self.add_noise

        # Termination condition
        if self.ep_length >= self.max_operations:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Max. milling operations crossed'}

        elif tool_wear > self.wear_threshold and np.random.uniform() < self.breakdown_chance:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Tool breakdown'}

        else:
            terminated = False
            info = {'action':'Continue'}

            reward = 0.0
            if tool_wear < self.wear_threshold:
                reward += self.R1*self.df_index
            else:
                # Threshold breached
                reward += self.R2*self.df_index # farther away from threshold => more penalty

            # Based on the action = 1 replace the tool or if 0, continue with normal operation
            if action:
                reward += self.R3
                # We replace the tool - so roll back tool life. -1 so that the increment in df_index will reset it to 0
                self.df_index = -1
                self.ep_tool_replaced += 1
                info = {'action':'Tool replaced'}

            # Increment reward for the episode based on final evaluation of reward of this step
            self.reward += float(reward/1e6)
            self.ep_total_reward += self.reward

            # Post process of step: Get next observation, fill history arrays
            self.ep_length += 1

            if self.df_index > (self.df_length-2):
                self.df_index = -1
                terminated = True
                info = {'data_index':'Data over'}

        # We can now read the next state, for agent's policy to predict the "Action"
        self.df_index += 1
        state_ = self._get_observation(self.df_index)

        return state_, self.reward, terminated, info

    def _get_observation(self, index):
        next_state = np.array([
            self.df['time'][index],
            self.df['tool_wear'][index]
        ], dtype=np.float32)

        return next_state

    def reset(self):
        # Append Episode stats. before resetting variables
        self.ep_rewards_history.append(self.ep_total_reward)
        self.ep_length_history.append(self.ep_length)
        self.ep_tool_replaced_history.append(self.ep_tool_replaced)

        # Reset environment variables and stats.
        self.df_index = 0
        self.reward = 0.0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_length = 0

        # Get the new state
        self.state = self._get_observation(self.df_index)
        terminated = False
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human', close=False):
        print(f'{self.df_index:>3d} | Ep.Len.: {self.ep_length:>3d} | Reward: {self.reward:>10.4f} | Wear: {wear:>5.4f} | {info}')

    def close(self):
        del [self.df]
        gc.collect()
        print('** -- Envoronment closed. Data-frame memory released. Garbage collector invoked successfully -- **')

class MillingTool_MS_V3(gym.Env):
    """Custom Milling Tool Wear Environment that follows the Open AI gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, wear_threshold, max_operations, add_noise, breakdown_chance, R1=1.0, R2=-1.0, R3=-100.0):
        print(f'** -- Multi-variate state V3 env. R1: {R1}, R2: {R2}, R3: {R3}. Noise: {add_noise}. Break-down chance: {breakdown_chance} -- **')

        # Machine data frame properties
        self.df = df
        self.df_length = len(self.df.index)
        self.df_index = 0

        # Milling operation and tool parameters
        self.wear_threshold = wear_threshold
        self.max_operations = max_operations
        self.breakdown_chance = breakdown_chance
        self.add_noise = add_noise
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.reward = 0.0

        # Statistics
        self.ep_length = 0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_rewards_history = []
        self.ep_length_history = []
        self.ep_tool_replaced_history = []

        ## Gym interface Obsercation and Action spaces
        # All features are normalized [0, 1]
        self.min_feature = 0.0
        self.max_feature = 1.0

        # Define state and action limits
        self.low_state = np.array([self.min_feature, self.min_feature, self.min_feature,
                                   self.min_feature, self.min_feature, self.min_feature,
                                   self.min_feature, self.min_feature], dtype=np.float32)

        self.high_state = np.array([self.max_feature, self.max_feature, self.max_feature,
                                    self.max_feature, self.max_feature, self.max_feature,
                                    self.max_feature, self.max_feature], dtype=np.float32)

        # Observation and action spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    def step(self, action):
        """
        Args: action. Discrete - 1=Replace tool, 0=Continue milling operation
        Returns: next_state, reward, terminated, truncated , info
        """
        # Get current observation from environment
        self.state = self._get_observation(self.df_index)
        tool_wear = self.state[0]

        # Add white noise for robustness
        # if self.add_noise:
        #     tool_wear += np.random.rand()/self.add_noise

        # Termination condition
        if self.ep_length >= self.max_operations:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Max. milling operations crossed'}

        elif tool_wear > self.wear_threshold and np.random.uniform() < self.breakdown_chance:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Tool breakdown'}

        else:
            terminated = False
            info = {'action':'Continue'}

            reward = 0.0
            if tool_wear < self.wear_threshold:
                reward += self.R1*self.df_index
            else:
                # Threshold breached
                reward += self.R2*self.df_index # farther away from threshold => more penalty

            # Based on the action = 1 replace the tool or if 0, continue with normal operation
            if action:
                reward += self.R3
                # We replace the tool - so roll back tool life. -1 so that the increment in df_index will reset it to 0
                self.df_index = -1
                self.ep_tool_replaced += 1
                info = {'action':'Tool replaced'}

            # Increment reward for the episode based on final evaluation of reward of this step
            self.reward += float(reward/1e6)
            self.ep_total_reward += self.reward

            # Post process of step: Get next observation, fill history arrays
            self.ep_length += 1

            if self.df_index > (self.df_length-2):
                self.df_index = -1
                info = {'data_index':'Data over'}

        # We can now read the next state, for agent's policy to predict the "Action"
        self.df_index += 1
        state_ = self._get_observation(self.df_index)

        return state_, self.reward, terminated, info

    # force_x; force_y; force_z; vibration_x; vibration_y; vibration_z; acoustic_emission_rms; tool_wear
    def _get_observation(self, index):
        next_state = np.array([
            self.df['tool_wear'][index],
            self.df['force_x'][index],
            self.df['force_y'][index],
            self.df['force_z'][index],
            self.df['vibration_x'][index],
            self.df['vibration_y'][index],
            self.df['vibration_z'][index],
            self.df['acoustic_emission_rms'][index]
        ], dtype=np.float32)

        return next_state

    def reset(self):
        # Append Episode stats. before resetting variables
        self.ep_rewards_history.append(self.ep_total_reward)
        self.ep_length_history.append(self.ep_length)
        self.ep_tool_replaced_history.append(self.ep_tool_replaced)

        # Reset environment variables and stats.
        self.df_index = 0
        self.reward = 0.0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_length = 0

        # Get the new state
        self.state = self._get_observation(self.df_index)
        terminated = False
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human', close=False):
        print(f'{self.df_index:>3d} | Ep.Len.: {self.ep_length:>3d} | Reward: {self.reward:>10.4f} | Wear: {wear:>5.4f} | {info}')

    def close(self):
        del [self.df]
        gc.collect()
        print('** -- Envoronment closed. Data-frame memory released. Garbage collector invoked successfully -- **')

## Single variable State V.2.0
class MillingTool_SS_V3(gym.Env):
    """Custom Milling Tool Wear Environment that follows the Open AI gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, wear_threshold, max_operations, add_noise, breakdown_chance, R1=1.0, R2=-1.0, R3=-100.0):
        print(f'\n** -- Simple single variable state V3 env.  R1: {R1}, R2: {R2}, R3: {R3}. Noise: {add_noise}. Break-down chance: {breakdown_chance} -- **')

        # Machine data frame properties
        self.df = df
        self.df_length = len(self.df.index)
        self.df_index = 0

        # Milling operation and tool parameters
        self.wear_threshold = wear_threshold
        self.max_operations = max_operations
        self.breakdown_chance = breakdown_chance
        self.add_noise = add_noise
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.reward = 0.0

        # Statistics
        self.ep_length = 0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_rewards_history = []
        self.ep_length_history = []
        self.ep_tool_replaced_history = []

        ## Gym interface Obsercation and Action spaces
        # All features are normalized [0, 1]
        self.min_feature = 0.0
        self.max_feature = 1.0

        # Define state and action limits
        self.low_state = np.array([self.min_feature, self.min_feature], dtype=np.float32)
        self.high_state = np.array([self.max_feature, self.max_feature], dtype=np.float32)

        # Observation and action spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    def step(self, action):
        """
        Args: action. Discrete - 1=Replace tool, 0=Continue milling operation
        Returns: next_state, reward, terminated, truncated , info
        """
        # Get current observation from environment
        self.state = self._get_observation(self.df_index)
        time_step = self.state[0]
        tool_wear = self.state[1]

        # Add white noise for robustness
        # if self.add_noise:
        #     tool_wear += np.random.rand()/self.add_noise

        # Termination condition
        if self.ep_length >= self.max_operations:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Max. milling operations crossed'}

        elif tool_wear > self.wear_threshold and np.random.uniform() < self.breakdown_chance:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Tool breakdown'}

        else:
            terminated = False
            info = {'action':'Continue'}

            reward = 0.0
            if tool_wear < self.wear_threshold:
                reward += self.R1*self.df_index
            else:
                # Threshold breached
                reward += self.R2*self.df_index # farther away from threshold => more penalty

            # Based on the action = 1 replace the tool or if 0, continue with normal operation
            if action:
                reward += self.R3
                # We replace the tool - so roll back tool life. -1 so that the increment in df_index will reset it to 0
                self.df_index = -1
                self.ep_tool_replaced += 1
                info = {'action':'Tool replaced'}

            # Increment reward for the episode based on final evaluation of reward of this step
            self.reward += float(reward/1e6)
            self.ep_total_reward += self.reward

            # Post process of step: Get next observation, fill history arrays
            self.ep_length += 1

            if self.df_index > (self.df_length-2):
                self.df_index = -1
                info = {'data_index':'Data over'}

        # We can now read the next state, for agent's policy to predict the "Action"
        self.df_index += 1
        state_ = self._get_observation(self.df_index)

        return state_, self.reward, terminated, info

    def _get_observation(self, index):
        next_state = np.array([
            self.df['time'][index],
            self.df['tool_wear'][index]
        ], dtype=np.float32)

        return next_state

    def reset(self):
        # Append Episode stats. before resetting variables
        self.ep_rewards_history.append(self.ep_total_reward)
        self.ep_length_history.append(self.ep_length)
        self.ep_tool_replaced_history.append(self.ep_tool_replaced)

        # Reset environment variables and stats.
        self.df_index = 0
        self.reward = 0.0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_length = 0

        # Get the new state
        self.state = self._get_observation(self.df_index)
        terminated = False
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human', close=False):
        print(f'{self.df_index:>3d} | Ep.Len.: {self.ep_length:>3d} | Reward: {self.reward:>10.4f} | Wear: {wear:>5.4f} | {info}')

    def close(self):
        del [self.df]
        gc.collect()
        print('** -- Envoronment closed. Data-frame memory released. Garbage collector invoked successfully -- **')

#####################################################################################################################################################
# VERSION 2.0
#####################################################################################################################################################


## V.4.0: MillingTool_MS_V2
## Add reward function elements to init method. R1, R2 and R3
class MillingTool_MS_V2(gym.Env):
    """Custom Milling Tool Wear Environment that follows the Open AI gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, wear_threshold, max_operations, add_noise, breakdown_chance, R1=1.0, R2=-1.0, R3=-100.0):
        print(f'\n** -- Multi-variate state V2 env. R1: {R1}, R2: {R2}, R3: {R3}. Noise: {add_noise}. Break-down chance: {breakdown_chance} -- **\n')

        # Machine data frame properties
        self.df = df
        self.df_length = len(self.df.index)
        self.df_index = 0

        # Milling operation and tool parameters
        self.wear_threshold = wear_threshold
        self.max_operations = max_operations
        self.breakdown_chance = breakdown_chance
        self.add_noise = add_noise
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.reward = 0.0

        # Statistics
        self.ep_length = 0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_rewards_history = []
        self.ep_length_history = []
        self.ep_tool_replaced_history = []

        ## Gym interface Obsercation and Action spaces
        # All features are normalized [0, 1]
        self.min_feature = 0.0
        self.max_feature = 1.0

        # Define state and action limits
        self.low_state = np.array([self.min_feature, self.min_feature, self.min_feature,
                                   self.min_feature, self.min_feature, self.min_feature,
                                   self.min_feature, self.min_feature], dtype=np.float32)

        self.high_state = np.array([self.max_feature, self.max_feature, self.max_feature,
                                    self.max_feature, self.max_feature, self.max_feature,
                                    self.max_feature, self.max_feature], dtype=np.float32)


        # Observation and action spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    def step(self, action):
        """
        Args: action. Discrete - 1=Replace tool, 0=Continue milling operation
        Returns: next_state, reward, terminated, truncated , info
        """
        # Get current observation from environment
        self.state = self._get_observation(self.df_index)
        tool_wear = self.state[0]

        # Add white noise for robustness
        if self.add_noise:
            tool_wear += np.random.rand()/self.add_noise

        # Termination condition
        if self.ep_length >= self.max_operations:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Max. milling operations crossed'}

        elif tool_wear > self.wear_threshold and np.random.uniform() < self.breakdown_chance:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Tool breakdown'}

        else:
            terminated = False
            info = {'action':'Continue'}

            reward = 0.0
            if tool_wear < self.wear_threshold:
                reward += self.R1*self.df_index
            else:
                # Threshold breached
                reward += self.R2*self.df_index # farther away from threshold => more penalty

            # Based on the action = 1 replace the tool or if 0, continue with normal operation
            if action:
                reward += self.R3
                # We replace the tool - so roll back tool life. -1 so that the increment in df_index will reset it to 0
                self.df_index = -1
                self.ep_tool_replaced += 1
                info = {'action':'Tool replaced'}

            # Increment reward for the episode based on final evaluation of reward of this step
            self.reward += float(reward/1e6)
            self.ep_total_reward += self.reward

            # Post process of step: Get next observation, fill history arrays
            self.ep_length += 1

            if self.df_index > (self.df_length-2):
                self.df_index = -1
                info = {'data_index':'Data over'}

        # We can now read the next state, for agent's policy to predict the "Action"
        self.df_index += 1
        state_ = self._get_observation(self.df_index)

        return state_, self.reward, terminated, info

    # force_x; force_y; force_z; vibration_x; vibration_y; vibration_z; acoustic_emission_rms; tool_wear
    def _get_observation(self, index):
        next_state = np.array([
            self.df['tool_wear'][index],
            self.df['force_x'][index],
            self.df['force_y'][index],
            self.df['force_z'][index],
            self.df['vibration_x'][index],
            self.df['vibration_y'][index],
            self.df['vibration_z'][index],
            self.df['acoustic_emission_rms'][index]
        ], dtype=np.float32)

        return next_state

    def reset(self):
        # Append Episode stats. before resetting variables
        self.ep_rewards_history.append(self.ep_total_reward)
        self.ep_length_history.append(self.ep_length)
        self.ep_tool_replaced_history.append(self.ep_tool_replaced)

        # Reset environment variables and stats.
        self.df_index = 0
        self.reward = 0.0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_length = 0

        # Get the new state
        self.state = self._get_observation(self.df_index)
        terminated = False
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human', close=False):
        print(f'{self.df_index:>3d} | Ep.Len.: {self.ep_length:>3d} | Reward: {self.reward:>10.4f} | Wear: {wear:>5.4f} | {info}')

    def close(self):
        del [self.df]
        gc.collect()
        print('** -- Envoronment closed. Data-frame memory released. Garbage collector invoked successfully -- **')


## Single variable State V.2.0
class MillingTool_SS_V2(gym.Env):
    """Custom Milling Tool Wear Environment that follows the Open AI gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, wear_threshold, max_operations, add_noise, breakdown_chance, R1=1.0, R2=-1.0, R3=-100.0):
        print(f'\n** -- Simple single variable state V2 env.  R1: {R1}, R2: {R2}, R3: {R3}. Noise: {add_noise}. Break-down chance: {breakdown_chance} -- **')

        # Machine data frame properties
        self.df = df
        self.df_length = len(self.df.index)
        self.df_index = 0

        # Milling operation and tool parameters
        self.wear_threshold = wear_threshold
        self.max_operations = max_operations
        self.breakdown_chance = breakdown_chance
        self.add_noise = add_noise
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.reward = 0.0

        # Statistics
        self.ep_length = 0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_rewards_history = []
        self.ep_length_history = []
        self.ep_tool_replaced_history = []

        ## Gym interface Obsercation and Action spaces
        # All features are normalized [0, 1]
        self.min_feature = 0.0
        self.max_feature = 1.0

        # Define state and action limits
        self.low_state = np.array([self.min_feature, self.min_feature], dtype=np.float32)
        self.high_state = np.array([self.max_feature, self.max_feature], dtype=np.float32)

        # Observation and action spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    def step(self, action):
        """
        Args: action. Discrete - 1=Replace tool, 0=Continue milling operation
        Returns: next_state, reward, terminated, truncated , info
        """
        # Get current observation from environment
        self.state = self._get_observation(self.df_index)
        time_step = self.state[0]
        tool_wear = self.state[1]

        # Add white noise for robustness
        if self.add_noise:
            tool_wear += np.random.rand()/self.add_noise

        # Termination condition
        if self.ep_length >= self.max_operations:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Max. milling operations crossed'}

        elif tool_wear > self.wear_threshold and np.random.uniform() < self.breakdown_chance:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Tool breakdown'}

        else:
            terminated = False
            info = {'action':'Continue'}

            reward = 0.0
            if tool_wear < self.wear_threshold:
                reward += self.df_index
            else:
                # Threshold breached
                reward += -self.df_index # farther away from threshold => more penalty

            # Based on the action = 1 replace the tool or if 0, continue with normal operation
            if action:
                reward += -100.0
                # We replace the tool - so roll back tool life. -1 so that the increment in df_index will reset it to 0
                self.df_index = -1
                self.ep_tool_replaced += 1
                info = {'action':'Tool replaced'}

            # Increment reward for the episode based on final evaluation of reward of this step
            self.reward += float(reward/1e6)
            self.ep_total_reward += self.reward

            # Post process of step: Get next observation, fill history arrays
            self.ep_length += 1

            if self.df_index > (self.df_length-2):
                self.df_index = -1
                info = {'data_index':'Data over'}

        # We can now read the next state, for agent's policy to predict the "Action"
        self.df_index += 1
        state_ = self._get_observation(self.df_index)

        return state_, self.reward, terminated, info

    def _get_observation(self, index):
        next_state = np.array([
            self.df['time'][index],
            self.df['tool_wear'][index]
        ], dtype=np.float32)

        return next_state

    def reset(self):
        # Append Episode stats. before resetting variables
        self.ep_rewards_history.append(self.ep_total_reward)
        self.ep_length_history.append(self.ep_length)
        self.ep_tool_replaced_history.append(self.ep_tool_replaced)

        # Reset environment variables and stats.
        self.df_index = 0
        self.reward = 0.0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_length = 0

        # Get the new state
        self.state = self._get_observation(self.df_index)
        terminated = False
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human', close=False):
        print(f'{self.df_index:>3d} | Ep.Len.: {self.ep_length:>3d} | Reward: {self.reward:>10.4f} | Wear: {wear:>5.4f} | {info}')

    def close(self):
        del [self.df]
        gc.collect()
        print('** -- Envoronment closed. Data-frame memory released. Garbage collector invoked successfully -- **')


#####################################################################################################################################################
# VERSION 1.0
#####################################################################################################################################################

## V3: Add multi-state capability
# force_x; force_y; force_z; vibration_x; vibration_y; vibration_z; acoustic_emission_rms; tool_wear; ACTION_CODE
class MillingTool_MS(gym.Env):
    """Custom Milling Tool Wear Environment that follows the Open AI gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, wear_threshold, max_operations, add_noise, breakdown_chance):
        print(f'\n** -- Multi-variate state. [V.4.0 +1.0*indx, -1.2*idx and -4.0] Noise: {add_noise}. Break-down chance: {breakdown_chance} -- **\n')

        # Machine data frame properties
        self.df = df
        self.df_length = len(self.df.index)
        self.df_index = 0

        # Milling operation and tool parameters
        self.wear_threshold = wear_threshold
        self.max_operations = max_operations
        self.breakdown_chance = breakdown_chance
        self.add_noise = add_noise
        self.reward = 0.0

        # Statistics
        self.ep_length = 0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_rewards_history = []
        self.ep_length_history = []
        self.ep_tool_replaced_history = []

        ## Gym interface Obsercation and Action spaces
        # All features are normalized [0, 1]
        self.min_feature = 0.0
        self.max_feature = 1.0

        # Define state and action limits
        self.low_state = np.array([self.min_feature, self.min_feature, self.min_feature,
                                   self.min_feature, self.min_feature, self.min_feature,
                                   self.min_feature, self.min_feature], dtype=np.float32)

        self.high_state = np.array([self.max_feature, self.max_feature, self.max_feature,
                                    self.max_feature, self.max_feature, self.max_feature,
                                    self.max_feature, self.max_feature], dtype=np.float32)


        # Observation and action spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    def step(self, action):
        """
        Args: action. Discrete - 1=Replace tool, 0=Continue milling operation
        Returns: next_state, reward, terminated, truncated , info
        """
        # Get current observation from environment
        self.state = self._get_observation(self.df_index)
        tool_wear = self.state[0]

        # Add white noise for robustness
        if self.add_noise:
            tool_wear += np.random.rand()/self.add_noise

        # Termination condition
        if self.ep_length >= self.max_operations:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Max. milling operations crossed'}

        elif tool_wear > self.wear_threshold and np.random.uniform() < self.breakdown_chance:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Tool breakdown'}

        else:
            terminated = False
            info = {'action':'Continue'}

            reward = 0.0
            if tool_wear < self.wear_threshold:
                reward += self.df_index
                # reward += 1.0
            else:
                # Threshold breached
                reward += -1.2*self.df_index # farther away from threshold => more penalty
                # reward += -4.0

            # Based on the action = 1 replace the tool or if 0, continue with normal operation
            if action:
                # V.3.3 reward += -10.0
                # reward -= 1.0*self.df_index
                reward += -4.0
                # reward += 0.5*abs(self.wear_threshold - tool_wear)
                # We replace the tool - so roll back tool life. -1 so that the increment in df_index will reset it to 0
                self.df_index = -1
                self.ep_tool_replaced += 1
                info = {'action':'Tool replaced'}

            # Increment reward for the episode based on final evaluation of reward of this step
            self.reward += float(reward/1e6)
            self.ep_total_reward += self.reward

            # Post process of step: Get next observation, fill history arrays
            self.ep_length += 1

            if self.df_index > (self.df_length-2):
                self.df_index = -1
                info = {'data_index':'Data over'}

        # We can now read the next state, for agent's policy to predict the "Action"
        self.df_index += 1
        state_ = self._get_observation(self.df_index)

        return state_, self.reward, terminated, info

    # force_x; force_y; force_z; vibration_x; vibration_y; vibration_z; acoustic_emission_rms; tool_wear
    def _get_observation(self, index):
        next_state = np.array([
            self.df['tool_wear'][index],
            self.df['force_x'][index],
            self.df['force_y'][index],
            self.df['force_z'][index],
            self.df['vibration_x'][index],
            self.df['vibration_y'][index],
            self.df['vibration_z'][index],
            self.df['acoustic_emission_rms'][index]
        ], dtype=np.float32)

        return next_state

    def reset(self):
        # Append Episode stats. before resetting variables
        self.ep_rewards_history.append(self.ep_total_reward)
        self.ep_length_history.append(self.ep_length)
        self.ep_tool_replaced_history.append(self.ep_tool_replaced)

        # Reset environment variables and stats.
        self.df_index = 0
        self.reward = 0.0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_length = 0

        # Get the new state
        self.state = self._get_observation(self.df_index)
        terminated = False
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human', close=False):
        print(f'{self.df_index:>3d} | Ep.Len.: {self.ep_length:>3d} | Reward: {self.reward:>10.4f} | Wear: {wear:>5.4f} | {info}')

    def close(self):
        del [self.df]
        gc.collect()
        print('** -- Envoronment closed. Data-frame memory released. Garbage collector invoked successfully -- **')



## V2: Add realistic situations
## 1. Random tool breakdown after crossing 30% of wear threshold
## 2. Random white noise added to wear

class MillingTool_SS(gym.Env):
    """Custom Milling Tool Wear Environment that follows the Open AI gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, wear_threshold, max_operations, add_noise, breakdown_chance):
        print(f'\n** -- Simple single variable state. Noise: {add_noise}. Break-down chance: {breakdown_chance} -- **')

        # Machine data frame properties
        self.df = df
        self.df_length = len(self.df.index)
        self.df_index = 0

        # Milling operation and tool parameters
        self.wear_threshold = wear_threshold
        self.max_operations = max_operations
        self.breakdown_chance = breakdown_chance
        self.add_noise = add_noise
        self.reward = 0.0

        # Statistics
        self.ep_length = 0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_rewards_history = []
        self.ep_length_history = []
        self.ep_tool_replaced_history = []

        ## Gym interface Obsercation and Action spaces
        # All features are normalized [0, 1]
        self.min_feature = 0.0
        self.max_feature = 1.0

        # Define state and action limits
        self.low_state = np.array([self.min_feature, self.min_feature], dtype=np.float32)
        self.high_state = np.array([self.max_feature, self.max_feature], dtype=np.float32)

        # Observation and action spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    def step(self, action):
        """
        Args: action. Discrete - 1=Replace tool, 0=Continue milling operation
        Returns: next_state, reward, terminated, truncated , info
        """
        # Get current observation from environment
        self.state = self._get_observation(self.df_index)
        time_step = self.state[0]
        tool_wear = self.state[1]

        # Add white noise for robustness
        if self.add_noise:
            tool_wear += np.random.rand()/self.add_noise

        # Termination condition
        if self.ep_length >= self.max_operations:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Max. milling operations crossed'}

        elif tool_wear > self.wear_threshold and np.random.uniform() < self.breakdown_chance:
            terminated = True
            self.reward = 0.0
            self.df_index = -1
            info = {'termination':'Tool breakdown'}

        else:
            terminated = False
            info = {'action':'Continue'}

            reward = 0.0
            if tool_wear < self.wear_threshold:
                reward += self.df_index
            else:
                # Threshold breached
                reward += -self.df_index # farther away from threshold => more penalty

            # Based on the action = 1 replace the tool or if 0, continue with normal operation
            if action:
                reward += -100.0
                # We replace the tool - so roll back tool life. -1 so that the increment in df_index will reset it to 0
                self.df_index = -1
                self.ep_tool_replaced += 1
                info = {'action':'Tool replaced'}

            # Increment reward for the episode based on final evaluation of reward of this step
            self.reward += float(reward/1e6)
            self.ep_total_reward += self.reward

            # Post process of step: Get next observation, fill history arrays
            self.ep_length += 1

            if self.df_index > (self.df_length-2):
                self.df_index = -1
                info = {'data_index':'Data over'}

        # We can now read the next state, for agent's policy to predict the "Action"
        self.df_index += 1
        state_ = self._get_observation(self.df_index)

        return state_, self.reward, terminated, info

    def _get_observation(self, index):
        next_state = np.array([
            self.df['time'][index],
            self.df['tool_wear'][index]
        ], dtype=np.float32)

        return next_state

    def reset(self):
        # Append Episode stats. before resetting variables
        self.ep_rewards_history.append(self.ep_total_reward)
        self.ep_length_history.append(self.ep_length)
        self.ep_tool_replaced_history.append(self.ep_tool_replaced)

        # Reset environment variables and stats.
        self.df_index = 0
        self.reward = 0.0
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_length = 0

        # Get the new state
        self.state = self._get_observation(self.df_index)
        terminated = False
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human', close=False):
        print(f'{self.df_index:>3d} | Ep.Len.: {self.ep_length:>3d} | Reward: {self.reward:>10.4f} | Wear: {wear:>5.4f} | {info}')

    def close(self):
        del [self.df]
        gc.collect()
        print('** -- Envoronment closed. Data-frame memory released. Garbage collector invoked successfully -- **')
