import math

base = open('py.log', 'r')
res = open('res.log', 'r')

line_num = 1
base_matrix = []

for line in base.readlines():
    words = line.split(' ')
    for word in words:
        number = word.replace('[', '')
        number = number.replace(']', '')
        number = number.strip()
        if len(number) > 0:
            try:
                x = float(number)
                base_matrix.append((x, line_num))
                # if len(base_matrix) < 10:
                #     print(x)
            except ValueError:
                pass

    line_num += 1

id = 0
line_num = 1
for line in res.readlines():
    words = line.split(' ')
    for word in words:
        number = word.replace('[', '')
        number = number.replace(']', '')
        number = number.strip()
        if len(number) > 0:
            # x = float(number)
            # print(x)
            try:
                x = float(number)
                # sprint(x)
                if abs(x - base_matrix[id][0]) > 1e-4:
                    print((x, line_num), base_matrix[id])
                    exit()
                id += 1
            except ValueError:
                pass

    line_num += 1
print("successfully match!")