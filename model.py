import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# color + P1 pieces + P2 pieces
BOARD_SHAPE = (1 + 6 + 6, 8, 8)
MOVES_SIZE = 1968

def encode(state):
    board = chess.Board(state)
    x = np.zeros(BOARD_SHAPE)

    # one plane encodes current color
    if board.turn:
        x[0, :, :] = 1.0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        # piece types numbered from 1
        plane = (piece.piece_type - 1) + 1

        # opponent's pieces in second set of planes
        if piece.color != board.turn:
            plane += 6

        row = square // 8
        column = square % 8

        x[plane, row, column] = 1.0
    return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class Model(nn.Module):
    def __init__(self, input_shape=BOARD_SHAPE, output_size=MOVES_SIZE):
        super(Model, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            Flatten())

        dummy = torch.zeros(1, *input_shape)
        flat_size = int(np.prod(self.body(dummy).size()))

        self.head_policy = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size))

        self.head_value = nn.Sequential(
            nn.Linear(flat_size, 1))

    def forward(self, x):
        features = self.body(x)
        policy = F.log_softmax(self.head_policy(features), dim=1)
        value = torch.tanh(self.head_value(features))
        return policy, value
