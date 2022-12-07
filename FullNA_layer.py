#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import micro_env
import numpy as np


# In[2]:


class FullNALayer:
    def __init__(self, num_devices, devices=None):
        if devices is not None:
            self.devices = devices
        else:
            self.devices = [micro_env.get_random_env() for _ in range(num_devices)]
        self.history_actions = []

        self.reset()
        for d in self.devices:
            print(f'schedule: {d.from_timeslotnumber} - {d.to_timeslotnumber}, duration: {d.consumption_period}')

    def get_action_shape(self):
        return len(self.devices)

    def reset(self):
        self.done = False
        self.time_stamp = 0
        for d in self.devices:
            d.reset()
        return self.get_obs()

    def action_space_sample(self):
        return [random.randint(0, 1) for _ in self.devices]

    def get_obs_shape(self):
        return np.shape(self.get_obs())

    def get_obs(self):
        obs = []
        for d in self.devices:
            obs.extend(d.get_obs())
        return np.array(obs)

    def reward(self, action):
        r = 0
        for d, a in zip(self.devices, action):
            r += d.reward(a)
        return r

    def step(self, action):
        self.history_actions.append(action)
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

    return FullNALayer(Device_id, Device_usage, Energy_charge, udc, from_timeslotnumber, to_timeslotnumber,
                       consumption_period, penalty, incentive)


# In[3]:


if __name__ == '__main__':
    # print(Env.time_stamp)
    env = FullNALayer(3)

    # for i in range(0, int(1e5)):
    #     ob, r, done = env.step(1)
    # if done:
    #     env.reset()
    # if i % 1000 == 0:
    #     print(np.mean(env.episode_rewards[-100:]))
    while not env.done:
        a = env.action_space_sample()
        ob, r, done, _ = env.step(a)
        print(f'action: {a}, reward: {r}, obs: {ob}')


# In[ ]:




