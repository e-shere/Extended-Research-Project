import chess
import random
from pathlib import Path
import numpy as np
import math
import torch
import model

MOVES = []

Ps = {}  # initial policy
Ns = {}  # how many times state was visited
Nsa = {} # how many times action from state taken
Qsa = {} # Q value for action in the state

# run on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample(policy):
    return np.random.choice(MOVES, p=policy)

def apply(state, action):
    board = chess.Board(state)
    board.push(chess.Move.from_uci(MOVES[action]))
    return board.fen()

class Agent():

    def __init__(self, previous=False):
        self.net = model.Model().to(device)
        filename = 'model-previous.dat' if previous else 'model.dat'
        if Path(filename).exists():
            w = torch.load(filename)
            self.net.load_state_dict(w)


    def predict(self, state):
        with torch.no_grad():
            state = model.encode(state)
            state = torch.FloatTensor(state).to(device)
            state = torch.unsqueeze(state, dim=0)
            policy, value = self.net(state)
            policy = policy.cpu()
            value = value.cpu()
            policy = torch.squeeze(policy, dim=0)
            policy = torch.exp(policy)
            policy = policy.numpy()
            value = torch.squeeze(value, dim=0).item()
        return policy, value

    def search(self, state):
        # these are updated during search
        global Ps
        global Ns
        global Qsa
        global Nsa

        CPUCT = 1.0
        EPS = 1e-6

        board = chess.Board(state)
        if board.is_game_over():
            return -1.0

        if board.fullmove_number > 100:
            return 0.0

        valid = set([m.uci() for m in board.legal_moves])
        valid = np.array([1.0 if m in valid else 0.0 for m in MOVES])

        # leaf node - initialise policy from NN prediction
        if state not in Ps:
            Ps[state], value = self.predict(state)
            Ps[state] = Ps[state] * valid  # mask invalid moves
            Ps[state] /= np.sum(Ps[state]) # renormalise
            Ns[state] = 0
            return value

        action = -1

        # select action with highest "upper confidence bound"
        for a in range(len(MOVES)):
            if valid[a] == 0.0:
                continue
            if (state, a) in Qsa:
                u = Qsa[(state, a)] + \
                    CPUCT * Ps[state][a] * math.sqrt(Ns[state]) / (1 + Nsa[(state, a)])
            else:
                u = CPUCT * Ps[state][a] * math.sqrt(Ns[state] + EPS)

            if action == -1 or u > best:
                best = u
                action = a
        assert(action != -1)

        # take action and unroll game further
        value = - self.search(apply(state, action))

        # update after search
        if (state, action) in Qsa:
            Qsa[(state, action)] = \
                (Nsa[(state, action)] * Qsa[(state, action)] + value) / (Nsa[(state, action)] + 1)
            Nsa[(state, action)] += 1
        else:
            Qsa[(state, action)] = value
            Nsa[(state, action)] = 1

        Ns[state] += 1
        return value

    def policy(self, state):
        for i in range(100):
            self.search(state)

        counts = [Nsa[(state, a)] if (state, a) in Nsa else 0
            for a in range(len(MOVES))]

        EXPLORATION = 1.0
        policy = [x ** (1.0 / EXPLORATION) for x in counts]
        policy /= np.sum(policy)
        return policy

def list_moves():
    shifts = \
        [(0, x) for x in range(-7, 8)] + \
        [(x, 0) for x in range(-7, 8)] + \
        [(x, x) for x in range(-7, 8)] +  \
        [(x, -x) for x in range(-7, 8)] + \
        [(x, y) for x in [-1, 1] for y in [-2, 2]] + \
        [(x, y) for x in [-2, 2] for y in [-1, 1]]

    moves = [(x, y, x + dx, y + dy)
        for x in range(8) for y in range (8)
        for (dx, dy) in shifts
        if (x + dx) in range(8) and (y + dy) in range(8) and
           (x, y) != (x + dx, y + dy)]

    promotions = [(x, y1, x + dx, y2)
        for x in range(8)
        for dx in range(-1, 2)
        for (y1, y2) in [(1, 0), (6, 7)]
        if (x + dx) in range(8)]

    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rows = ['1', '2', '3', '4', '5', '6', '7', '8']
    figures = ['q', 'r', 'b', 'n']

    ucis = \
        [columns[x1] + rows[y1] + columns[x2] + rows[y2]
            for (x1, y1, x2, y2) in moves] + \
        [columns[x1] + rows[y1] + columns[x2] + rows[y2] + f
            for (x1, y1, x2, y2) in promotions for f in figures]

    return list(sorted(ucis))

MOVES = list_moves()
