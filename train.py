#!/usr/bin/python3
import sys
import chess
import numpy as np
from pathlib import Path
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import model
import gzip

BATCH_SIZE = 100

# read all games
games = collections.deque()
with gzip.open(sys.stdin.buffer, mode='rt') as lines:
    for line in lines:
        state, value, policy = line.rstrip('\n').split('\t')
        state = model.encode(state)
        value = np.float32(value)
        policy = np.array([np.float32(x) for x in policy.split(' ')])
        games.append((state, value, policy))

def sample(size):
    idx = np.random.choice(len(games), size, replace=True)
    states, values, policies = zip(*[games[i] for i in idx])
    return np.array(states), np.array(values), np.array(policies)

def calc_loss(batch, net):
    states, values, policies = batch
    states = torch.FloatTensor(states)
    values_target = torch.FloatTensor(values)
    values_target = torch.unsqueeze(values_target, dim=1)
    policies_target = torch.FloatTensor(policies)

    policies_estimate, values_estimate = net(states)

    return nn.KLDivLoss()(policies_estimate, policies_target) + \
        nn.MSELoss()(values_estimate, values_target)

net = model.Model()

if Path('model.dat').exists():
    w = torch.load('model.dat')
    net.load_state_dict(w)

optimiser = optim.Adam(net.parameters(), lr=1e-3)
f = open("trainresults.txt", "a+")
for batch_index in range(500):
    optimiser.zero_grad()
    batch = sample(BATCH_SIZE)
    loss = calc_loss(batch, net)
    loss.backward()
    optimiser.step()

    print(f"{batch_index}\t{loss.item():.6f}")
    f.write(f"{loss.item():.6f}" + "\n")
f.close()
torch.save(net.state_dict(), "model.dat")
