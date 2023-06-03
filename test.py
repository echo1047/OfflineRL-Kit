import gym
import torch
import numpy as np


env = gym.make('Swimmer-v3')
env.reset()

tr = torch.tensor([1, 1]).to('cuda')
tr = tr.detach()
tr.numpy()
t = np.array(tr)
ns, r, d, _ = env.step(t)
print(ns)