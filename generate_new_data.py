from agents import *

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print('Using GPU!')
GAME_SIZE = 4
SCORE_TO_WIN = 2048
EPISODE = 4000
game_train = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
agent = ConvAgent(game_train)
agent2 = ExpectiMaxAgent(game_train)
total_score = 0
index = 0
txt_dir = "./dataset_new2/data0.txt"
file = open(txt_dir, mode='w')

for ep in range(EPISODE):
    _ = game_train.reset()
    if ep % 10 == 0 and ep != 0:
        index += 1
        txt_dir = "./dataset_new2/data" + str(index) + ".txt"
        file.close()
        file = open(txt_dir, mode='w')
    while game_train.end == 0:
        state = game_train.board
        max_score = np.max(state)
        if max_score <= 64:
            action = agent.step()
            game_train.move(action)
        else:
            state_print = np.reshape(state, [1, 16]).squeeze()
            action = agent2.step()
            game_train.move(action)
            for _ in range(4):
                print(state_print, " ", action, file=file)
                state = np.rot90(state)
                action = (action + 1) % 4
                state_print = np.reshape(state, [1, 16]).squeeze()