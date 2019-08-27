#!/usr/bin/python3
import sys
import chess
import random

import common

# Obtain random endgame position
positions = open('mates_and_endgames.txt').read().splitlines()
state = random.choice(positions)

agent = common.Agent()
experience = []

# Play out position until the game is over, or more than 50 moves were made
while True:
    board = chess.Board(state)
    if board.is_game_over() or board.fullmove_number > 50:
        break
    policy = agent.policy (state)
    experience.append((state, policy))
    move = common.sample (policy)
    board.push(chess.Move.from_uci(move))
    state = board.fen()
    sys.stderr.write(state + "\n")

# Assign game result
board = chess.Board(state)
results = [-0.75, -0.75]
if board.is_checkmate():
    if board.turn:
        results = [+1, -1] # black won
    else:
        results = [-1, +1] # white won

# Append for later training
for item in experience:
    fen = item[0]
    policy = item[1]
    turn = 1 if chess.Board(fen).turn else 0 # white is 1 black is 0
    value = results[turn]
    line = fen + '\t' + str(value) + '\t' + ' '.join([format(x, '.6f') for x in policy])
    print(line)
