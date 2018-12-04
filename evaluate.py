#from game2048.game import Game
#from game2048.displays import Display
from game import Game
from displays import Display
import numpy as np


def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    #agent = AgentClass(game, display=Display(), **kwargs)
    agent = AgentClass(game, display=Display(), **kwargs)
    #agent.play(verbose=True)
    _ = game.reset()
    while game.end == 0:
        state = game.board
        state = np.reshape(state, [1, 16]).squeeze()
        action = agent.step()
        game.move(action)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 10

    from agents import ConvAgent

    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=ConvAgent)
        scores.append(score)

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
