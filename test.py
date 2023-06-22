import csv
import regex as re
with open('vicuna-7b-with-explanasion-correct.csv', 'r', encoding='utf8') as f:
    lines = csv.reader(f, delimiter=',')

triple = "['It\'s_Great_to_Be_Young | editor | MaxBenedict']"
new = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
new = re.sub(r'_', ' ', new).lower()  # Lowercase, replace _ with space
new = re.sub(r'\s+', ' ', new).lower()
print(new)
adjusttriple = new.split(' | ')
print(adjusttriple)
manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
print(manualmodified)