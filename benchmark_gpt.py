import requests
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
import openai

# Fill these in, for security reasons left blank
api_key = 'sk-'
org_id = 'org-'

openai.organization = org_id
openai.api_key = api_key

web = pd.DataFrame(load_dataset('UofA-LINGO/text_to_triplets_new_ins')['test'])  # Webnlg test set
web = web.assign(instruction='Pretend that you are great at parsing text and can effectively extract all entities and their relationships from a text. I will give you some text and you will extract all the possible triples. Please extract all the triples using the style of DBpedia vocabulary where possible, although prefixes should not be added. Use literal values where appropriate. Ensure that every possible relation is extracted. The output format is: (subject | relationship | object)')
web = web.assign(original_output='')
web = web.rename(columns={"instruction": "instruction", "input": "input", "output": "ground_truth", "original_output": "response"})
print(web.head())
responses = []
try:
    for i in range(len(web['input'])):
        # Setup prompt for gpt-4
        mess=[{"role": "system", "content": web['instruction'][i]}]
        mess.append({"role": "user", "content": web['input'][i]})
    
        # Uncomment to send to gpt-4. Good idea to leave commented to avoid sending anything by accident.
        resp = openai.ChatCompletion.create(model='gpt-4', messages=mess, max_tokens=1024)
        responses.append(resp.choices[0].message.content)
        web['response'][i] = resp.choices[0].message.content
        print(web['response'][i])
        print(f"{i+1} / {len(web['input'])}")

        # Save every 100 iterations in case of errors
        if i % 100 == 0:
            web.to_csv(f'gpt4_results/gpt4-results-{i}.csv')
except Exception as e:
    print(e)
web.to_csv('gpt4-results-full.csv')
web_for_hf = DatasetDict({"test":Dataset.from_pandas(web)})
web_for_hf.push_to_hub('UofA-LINGO/gpt4-results-full', private=True)
