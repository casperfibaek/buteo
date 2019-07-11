import pandas as pd
from glob import glob


base = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\csv\\'
paths = glob(f"{base}training_material_vector_01*.csv")

new_csv = []

first_csv = open(paths[0], 'r').readlines()
for line in first_csv:
    new_csv.append(line.strip())

for path in paths[1:]:
    csv = open(path, 'r').readlines()

    for i, line in enumerate(csv):
        readied = ',' + ','.join(line.strip().split(',')[1:])
        new_csv[i] = new_csv[i] + readied

out_csv = open(f'{base}training_material_vector_01.csv', 'w')
out_csv.write('\n'.join(new_csv))
out_csv.close()
