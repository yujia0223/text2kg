import csv
with open('vicuna-7b-with-explanasion-correct.csv', 'r', encoding='utf8') as f:
    lines = csv.reader(f, delimiter=',')