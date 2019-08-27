#!/usr/bin/python3
import sys
import chess

import common

state = chess.STARTING_FEN
agent = common.Agent()

def react(command):
    global state
    if command == 'uci':
        return [ 'id name My Chess', 'id name Me', 'uciok' ]
    elif command == 'isready':
        return ['readyok']
    elif command.startswith('position fen'):
        state = command[len('position fen'):]
        return []
    elif command == 'position startpos':
        state = chess.STARTING_FEN
        return []
    elif command.startswith('position startpos moves '):
        moves = command[len('position startpos moves '):]
        moves = moves.split(' ')
        board = chess.Board()
        for move in moves:
            board.push(chess.Move.from_uci(move))
        state = board.fen()
        return []
    elif command == 'stop' or command.startswith('go movetime'):
        policy = agent.policy (state)
        move = common.sample (policy)
        return ['bestmove ' + move]
    elif command.startswith('go infinite'):
        policy = agent.policy (state)
        move = common.sample (policy)
        return ['info depth 1', 'info depth 1 score cp 0 pv ' + move]
    elif command == 'quit':
        sys.exit(0)
    else:
        return []

def test_react():
    assert react('uci')[-1] == 'uciok'

if __name__ == "__main__":
    with open('log.txt', 'w', 1) as log:
        while True:
            command = input()
            log.write('COMMAND: ' + command + '\n')
            responses = react(command)
            for response in responses:
                log.write('RESPONSE: ' + response + '\n')
                print(response)
