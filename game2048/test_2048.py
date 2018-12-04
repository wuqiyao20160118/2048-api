from agents import *

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print('Using GPU!')
GAME_SIZE = 4
SCORE_TO_WIN = 2048
EPISODE = 1000
game_test = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
agent = ConvAgent(game_test)
total_score = 0
file = open("./Conv.txt", "w")
for ep in range(EPISODE):
    _ = game_test.reset()
    while game_test.end == 0:
        state = game_test.board
        max_score = np.max(state)
        state = np.reshape(state, [1, 16]).squeeze()
        action = agent.step()
        game_test.move(action)
    state = game_test.board
    max_score = np.max(state)
    total_score += max_score
    print("Episode {}: {}".format(ep, max_score), file=file)

average_score = total_score / EPISODE
print("Average score is ", average_score, ".", file=file)