import pandas as pd
import pickle

gpt4 = pd.read_table("C:/Users/tyms4/OneDrive/Documents/Work/UofAResearch/scripts/raw_outputs/20230505-224811_web_nlg_test_50_samples_with_seed_66_num_of_runs_1_gpt4.tsv")
gpt4_0314 = pd.read_table("C:/Users/tyms4/OneDrive/Documents/Work/UofAResearch/scripts/raw_outputs/20230509-154059_web_nlg_test_50_samples_with_seed_66_num_of_runs_1_gpt4-0314.tsv")
gpt35 = pd.read_table("C:/Users/tyms4/OneDrive/Documents/Work/UofAResearch/scripts/raw_outputs/20231011-225458_web_nlg_test_all_samples_with_seed_66_num_of_runs_1_chatgpt35.tsv")

gpt4_new = pd.read_csv("C:/Users/tyms4/OneDrive/Documents/Work/UofAResearch/scripts/gpt4_results/gpt4-results-full.csv")

out4 = {}
out40341 = {}
out35 = {}
out4_new = {}
ind = 0

for entry in gpt4_new['response']:
    out4_new[ind] = entry
    ind += 1

"""
print(len(gpt35['triples']))
ind = 0
for val in gpt35['id']:
    if int(val.strip("Id"))-1 != ind:
        print(ind)
    ind += 1

ind = 0
for val in gpt35['triples']:
    val = val.strip('[').strip(']').split('), ')
    #print(val)
    for i in range(len(val)):
        if val[i][-1] != ')':
            val[i] += ')'
        val[i] = val[i].replace("', '", " | ").replace("'", "").strip()
    out35[ind] = ["\n".join(val)]
    ind+=1

print(out35[0])
with open("C:/Users/tyms4/OneDrive/Documents/Work/UofAResearch/scripts/raw_outputs/gpt35-results.pickle", "wb") as handle:
    pickle.dump(out35, handle, protocol=pickle.HIGHEST_PROTOCOL)

ind = 0
for val in gpt4['triples']:
    val = val.strip('[').strip(']').split('), ')
    for i in range(len(val)):
        if val[i][-1] != ')':
            val[i] += ')'
        val[i] = val[i].replace("', '", " | ").replace("'", "").strip()
    out4[ind] = val
    ind+=1

ind = 0
for val in gpt4_0314['triples']:
    val = val.strip('[').strip(']').split('), ')
    for i in range(len(val)):
        if val[i][-1] != ')':
            val[i] += ')'
        val[i] = val[i].replace("', '", " | ").replace("'", "").strip()
    out40341[ind] = val
    ind+=1


print(out4[0])

#with open("C:/Users/tyms4/OneDrive/Documents/Work/UofAResearch/scripts/raw_outputs/gpt4-results.pickle", "wb") as handle:
    #pickle.dump(out4, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open("C:/Users/tyms4/OneDrive/Documents/Work/UofAResearch/scripts/raw_outputs/gpt4-0314-results.pickle", "wb") as handle:
    #pickle.dump(out40341, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
with open("C:/Users/tyms4/OneDrive/Documents/Work/UofAResearch/scripts/raw_outputs/gpt4-results-full.pickle", "wb") as handle:
    pickle.dump(out4_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
gpt4inds = []
gpt4_0314inds = []
for val in gpt4['id']:
    gpt4inds.append(int(val.strip("Id"))-1)

for val in gpt4_0314['id']:
    gpt4_0314inds.append(int(val.strip("Id"))-1)

if gpt4_0314inds == gpt4inds:
    print("Equal")

print(gpt4inds)
print(len(gpt4inds))
"""
t = pd.read_pickle("C:/Users/tyms4/OneDrive/Documents/Work/UofAResearch/scripts/raw_outputs/gpt4-results-full.pickle").values()
print([x for x in t])
