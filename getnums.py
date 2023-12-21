lines = []
with open('webnlg-modified.txt', encoding='utf-8') as f:
    lines = f.readlines()

new = []
read = False
for l in lines:
    if l.strip() == 'Samples modified:':
        read = True
    if l.strip() == 'Test Samples Modified:':
        read = False
    if read:
        if l.strip().isnumeric():
            new.append(int(l.strip()))
print(new)
print(len(new))