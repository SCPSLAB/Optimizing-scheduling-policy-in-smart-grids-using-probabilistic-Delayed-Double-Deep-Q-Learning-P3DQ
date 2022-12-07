#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np


# In[13]:


class Env:
    def __init__(self, Device_id, Device_usage, Energy_charge, udc, from_timeslotnumber, to_timeslotnumber
                 , consumption_period, penalty, incentive):
        self.Device_id = Device_id
        self.Device_usage = Device_usage
        self.Energy_charge = Energy_charge
        self.udc = udc
        self.from_timeslotnumber = from_timeslotnumber
        self.to_timeslotnumber = to_timeslotnumber
        self.consumption_period = consumption_period
        self.penalty = penalty
        self.incentive = incentive
        self.preferences_satisfied = True
        self.done = False
        self.time_stamp = 0
        self.state_accumulation = 0
        self.episode_rewards = []
        self.history_actions = []
        print(f'schedule: {self.from_timeslotnumber} - {self.to_timeslotnumber}, duration: {self.consumption_period}')

    def get_action_shape(self):
        return 1

    def reset(self):
        self.done = False
        self.time_stamp = 0
        self.state_accumulation = 0
        self.history_actions = []
        return self.get_obs()

    def action_space_sample(self):
        return [random.randint(0, 1)]

    def get_obs_shape(self):
        return np.shape(self.get_obs())

    def get_obs(self):
        return [self.time_stamp, self.state_accumulation, self.from_timeslotnumber, self.consumption_period, self.to_timeslotnumber]

    def reward(self, action):
        under_schedule= self.from_timeslotnumber <= self.time_stamp < self.to_timeslotnumber
        at_to_timeslotnumber = self.time_stamp == self.to_timeslotnumber

        reward_function = (1 - under_schedule) *                           (at_to_timeslotnumber * self.incentive * (self.consumption_period -                                                                 np.abs(self.consumption_period - self.state_accumulation)) +                            action * self.penalty +
                           (1 - action) * self.incentive) + \
                          under_schedule* (
                                  action * (self.Energy_charge[self.time_stamp] * self.Device_usage) + \
                                  (1 - action) * self.udc * self.Device_usage)
        reward_function *= -1
        self.episode_rewards.append(reward_function)
        return reward_function

    def old_reward(self, action):
        under_schedule= self.from_timeslotnumber <= self.time_stamp <= self.to_timeslotnumber
        at_to_timeslotnumber = self.time_stamp == self.to_timeslotnumber

        reward_function = (1 - under_schedule) *                           (action * self.penalty +
                           (1 - action) * self.incentive) + \
                          under_schedule* (
                                  at_to_timeslotnumber * ((not self.preferences_satisfied) *
                                                      self.penalty *
                                                      np.abs(self.consumption_period - self.state_accumulation) +
                                                      self.preferences_satisfied *
                                                      self.incentive * self.consumption_period) + \
                                  (1 - at_to_timeslotnumber) *
                                  (action * self.Energy_charge[self.time_stamp] * self.Device_usage + \
                                   (1 - action) * self.udc * self.Device_usage))
        reward_function *= -1
        self.episode_rewards.append(reward_function)
        return reward_function

    def step(self, action):
        # Compatible to multiple device
#         action = action[0]

        self.history_actions.append(action)
        self.state_accumulation += action
        if self.state_accumulation != self.consumption_period and self.time_stamp == self.to_timeslotnumber:
            self.preferences_satisfied = False
        reward = self.reward(action)
        self.time_stamp += 1
        self.done = self.time_stamp == 24
        return self.get_obs(), reward, self.done, None


def get_random_env():
    Device_id = 1
    udc = 0.
    Energy_charge = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5])
    # Device_usage = np.random.randint(1, 10) / 10
    # from_timeslotnumber = np.random.randint(0, 12)
    # to_timeslotnumber = np.random.randint(13, 23)
    # consumption_period = np.random.randint(0, to_timeslotnumber - from_timeslotnumber)
    Device_usage = 1
    from_timeslotnumber = 5
    to_timeslotnumber = 20
    consumption_period = 4
    penalty = 10.
    incentive = -10.

    return Env(Device_id, Device_usage, Energy_charge, udc, from_timeslotnumber, to_timeslotnumber,
                       consumption_period, penalty, incentive)


# In[14]:


if __name__ == '__main__':
    env = get_random_env()
    rewards = []
    while not env.done:
        a = np.random.randint(0, 2)
        ob, r, done, _ = env.step(a)
        rewards.append(r)
        print(f'action: {a}, reward: {r}, obs: {ob}')
    print("Mean reward of random actions: %s " % (sum(rewards) / len(rewards)))


# In[ ]:




