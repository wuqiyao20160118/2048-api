txt = "./Conv.txt"

file = open(txt, 'r')
lines = file.readlines()
data_size = len(lines)
state = []
action = []
index = 0
score = 0
for index, line in enumerate(lines):
    line = line.strip('\n')
    line = line.split(':')
    line[1] = line[1].lstrip()
    score += float(line[1])
average_score = score / (index+1)
print(average_score)
