from agents import *

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print('Using GPU!')
GAME_SIZE = 4
SCORE_TO_WIN = 2048
EPISODE = 4000
game_test = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
agent = ConvAgent(game_test)
total_score = 0
txt_dir = "./my_data/data0.txt"
index = 0
file = open(txt_dir, "w")

for ep in range(EPISODE):
    _ = game_test.reset()
    if ep % 10 == 0 and ep != 0:
        index += 1
        txt_dir = "./my_data/data" + str(index) + ".txt"
        file.close()
        file = open(txt_dir, "w")
    while game_test.end == 0:
        state = game_test.board
        state_print = np.reshape(state, [1, 16]).squeeze()
        max_score = np.max(state)
        state = np.reshape(state, [1, 16]).squeeze()
        action = agent.step()
        game_test.move(action)
        print(state_print, file=file)
    # state = game_test.board
    # max_score = np.max(state)
    # total_score += max_score
    # print("Episode {}: {}".format(ep, max_score), file=file)
