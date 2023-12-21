import json

#f = open('C:/Users/tyms4/OneDrive/Documents/Work/UofAResearch/scripts/detailed_results_few_doc/orca-mini-3-7b-rf-docred-doc-det.json', encoding='utf-8')
f = open('C:/Users/tyms4/OneDrive/Documents/Work/UofAResearch/scripts/detailed_results_few_doc/Llama-2-13b-rf-docred-web-det.json', encoding='utf-8')
data = json.load(f)
f.close()

fields = ['id', 'text', 'ref', 'cand', 'triple_score', 'combination', 'triple_score_sum']

start = int(input("Enter starting number: "))

for i in data['id']:
    if i < start:
        continue
    print(f'{i}, {data["text"][i]}')
    print("Ref:", data['ref'][i])
    print("Cand:", data['cand'][i])
    #print("Score:", data['triple_score'][i])
    a = input()
    if a in ['e', 'exit', 'E', 'Exit']:
        break