# Benchmark imports
import pandas as pd  # Import pandas library
import sys
from datasets import load_dataset
import fire
import torch
import transformers
import csv
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import pickle
# alpaca-lora utils
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

from tqdm import tqdm

# Evaluation imports
# import warnings filter
from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning
# ignore all UndefinedMetricWarning warnings
simplefilter(action='ignore', category=UndefinedMetricWarning)
from bs4 import BeautifulSoup
import os
import regex as re
import itertools
import statistics
import sys
from nervaluate import Evaluator
import nltk
from nltk.util import ngrams
import string
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import preprocessing
import json

import ast
import numpy as np
import timeit

device = "cuda"
currentpath = os.getcwd()

def benchmark(
    model_path: str = "",
    tok: str = "",
    max_tokens: int = 1024,
    dump: str = "output.pickle",
    load_8bit: bool = False,
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    csv_file: str = None,  # New argument for CSV file
):
    if model_path == "":
        print("Enter the path to the model. (python benchmark_and_evaluate.py --model_path=/home/tsadler/models/vicuna-7b)")
        exit()
    prompter = Prompter(prompt_template)
    print(f"Benchmarking model at: {model_path}")
    print(f"Using tokenizer at (blank means model_path): {tok}")
    # tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    # tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer = None
    if tok == "":
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(tok)

    model = LlamaForCausalLM.from_pretrained(
        #"/home/taesiri/src/alpaca-lora/vicuna-7b--based-export-text-to-triplets-explanation-v3/",
        model_path,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=max_tokens,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        
        eos_tokens = [tokenizer.eos_token_id, tokenizer.encode("<s><|system|>")[-1], tokenizer.encode("<s>")[-1], tokenizer.encode("<|system|>")[-1], tokenizer.encode("<s>[INST]")[-1], tokenizer.encode("<<SYS>>")[-1], tokenizer.encode(")<")[-1]]

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(Stream(callback_func=callback))
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    dt = load_dataset("UofA-LINGO/text_to_triplets")
    output = {}
    for i in tqdm(range(0,25)):#len(dt["test"]))):
        entry = dt["test"][i]
        output[i] = list(evaluate(entry["instruction"], entry["context"]))
        print(output[i])
    
    with open(dump, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TSadler: Removing intermediate CSV file for combined code
    # generate dataframe for the evaluation code
    dt = load_dataset("UofA-LINGO/text_to_triplets")
    df = pd.DataFrame(dt["test"][0:25])
    df["gt"] = df["response"]
    df = df.drop(columns=["response"])
    df["model_output"] = [x[0] for x in output.values()]
    return df
    #df.to_csv("vicuna-7b-with-explanasion-correct.csv", index=False)

    # dump df as pickle
    #with open("vicuna-7b-with-explanasion-correct-df.pickle", "wb") as handle:
    #    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)



def getCandsAndRefsFromCsv(df):
    #df = pd.read_csv(filepath, header=0)
    print(df.head())

    allcand_ids = df.index.values
    all_text = df['context'].values

    all_cand_triples = []
    all_ref_triples = []
    for i in range(len(df)):
        triples_str_cand = df['model_output'].values[i]
        #DEBUG: print(triples_str_cand)
        # Remove EOS token
        triples_str_cand = triples_str_cand.replace('</s>', '')

        #triples_cand = re.findall(r"'(.*?)'", triples_str_cand)
        #triples_cand = re.findall(r"\((.*?)\)[<\n]", triples_str_cand)

        # New style triples
        triples_cand = triples_str_cand.strip().split('\n')

        #DEBUG: print('\n')
        #DEBUG: print(triples_cand)

        # Old style triples
        # exp_target = "Therefore, here is the answer in the correct format:"
        # Check for explanation-based model:
        # if triples_str_cand.find(exp_target) == -1:
        #     print("Found one with diff final answer prompt.")
        # else:
            # Only look at final output triples.
        #     triples_str_cand = triples_str_cand[triples_str_cand.find(exp_target)+len(exp_target):].strip()
        # This looks for the form of '...|...|...', which we expect our triples to be in. This must be followed
        # with a closing square bracket or comma to match to avoid edge cases such as the one below:
        # ['prop's | pred | value's'] would match to -> ['s | pred | value'] without the [],].
        # triples_cand_tmp = re.findall(r"'(.*?[|].*?[|].*?)'[],]", triples_str_cand)
        # triples_cand = []
        # for triple in triples_cand_tmp:
        #     triple = triple.split(' | ')
        #     triples_cand.append(f'({triple[0]}, {triple[1]}, {triple[2]})')

        tmp = []
        for triple in triples_cand:
            # Do not penalize the model for errors in splitting that cause empty strings
            if triple == '':
                continue
            # To prevent index errors later, pad incomplete triples with empty strings.
            if len(triple.split(', ')) < 3:
                if len(triple.split(', ')) == 1:
                    triple += ', , )'
                elif len(triple.split(', ')) == 2:
                    triple += ', )'
            if len(triple.split(', ')) > 3:
                print(triple)
                print(triple.split(', '))
            else:
                tmp.append(triple)
        all_cand_triples.append(tmp)

        triples_str_ref = df['gt'].values[i]
        triples_ref = []
        # Convert triples to new format, easier to compare later on than converting the other way
        for triple in ast.literal_eval("[" + triples_str_ref + "]")[0]:
            triple = triple.split(' | ')
            triples_ref.append(f'({triple[0]}, {triple[1]}, {triple[2]})')
        #DEBUG: print(triples_str_ref)
        #DEBUG: print(triples_ref)
        all_ref_triples.append(triples_ref)

    new_cand_list = []
    
    # This breaks our candidates
    for entry in all_cand_triples:
        new_triples = []
        for triple in entry:
            # Split camel case words into multiple words.
            new_triple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            # Tyler Sadler: Personally, I don't agree with replacing these. If we want the model to
            # follow a consistent output structure, it should be evaluated on underscores vs spaces.
            # Webnlg training data heavily weighted towards underscores, as that is the dbpedia convention.
            # new_triple = re.sub(r'_', ' ', new_triple).lower()
            # Multiple spaces to single space
            new_triple = re.sub(r'\s+', ' ', new_triple).lower()
            # adjusttriple = new_triple.split(' | ')
            # Removes bracketed terms, again I don't think this should run as it removes a potential source of
            # disagreement between the model and ground truth.
            # manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            # if manualmodified:
            #     adjusttriple[-1] = manualmodified.group(1)
            #     new_triple = ' | '.join(adjusttriple)
            new_triples.append(new_triple)
        new_cand_list.append(new_triples)
    #DEBUG: print(new_cand_list)

    new_ref_list = []
    for entry in all_ref_triples:
        new_triples = []
        # Same rationale as above applied here to remove parts of this.
        for triple in entry:
            new_triple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            # new_triple = re.sub(r'_', ' ', new_triple).lower()
            new_triple = re.sub(r'\s+', ' ', new_triple).lower()
            # adjusttriple = new_triple.split(' | ')
            # manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            # if manualmodified:
            #     adjusttriple[-1] = manualmodified.group(1)
            #     new_triple = ' | '.join(adjusttriple)
            new_triples.append(new_triple)
        new_ref_list.append(new_triples)

    return allcand_ids, all_text, all_cand_triples, new_cand_list, all_ref_triples, new_ref_list

def getRefs(filepath, allcand_ids):
    with open(filepath, encoding='utf-8') as fp:
        refssoup = BeautifulSoup(fp, 'lxml')

    refsentries = refssoup.find('benchmark').find('entries').find_all('entry')

    all_ref_triples = []
    for index in allcand_ids:
        id = int(index.split('Id')[1])-1
        entry = refsentries[id]
        entryreftriples = []
        modtriplesref = entry.find('modifiedtripleset').find_all('mtriple')
        for modtriple in modtriplesref:
            entryreftriples.append(modtriple.text)
        all_ref_triples.append(entryreftriples)

    new_ref_list = []

    for entry in all_ref_triples:
        new_triples = []
        for triple in entry:
            new_triple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            new_triple = re.sub(r'_', ' ', new_triple).lower()
            new_triple = re.sub(r'\s+', ' ', new_triple).lower()
            adjusttriple = new_triple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                new_triple = ' | '.join(adjusttriple)
            new_triples.append(new_triple)
        new_ref_list.append(new_triples)

    return all_ref_triples, new_ref_list

def getCandsFromRebelTsv(filepath):
    df = pd.read_csv(filepath, sep='\t', header=0)
    print(df.head())
    # df = df[:10]
    # df = df.sort_values(by=['id'])
    # print(df.head())
    # Get the triples for row with id 'Id770'
    # Example of triples: [('Abraham A. Ribicoff', 'born in', 'United States'), ('United States', 'has ethnic group', 'African Americans')]
    # triples_str_ref = df[df['id'] == 'Id770']['triples'].values[0]
    # # Convert the triples string to a list of tuples
    # triples = ast.literal_eval("[" + triples_str + "]")[0]
    allcand_ids = df['id'].values
    all_text = df['lexs'].values
    # from IPython import embed; embed()
    all_cand_triples = []
    for i in range(len(df)):
        # new_triples = []
        triples_str = df['triples'].values[i]
        triples = ast.literal_eval("[" + triples_str + "]")[0]
        # for triple in triples:
        #     triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
        #     new_triples.append(triple_str)
        all_cand_triples.append(triples)

    new_cand_list = []
    
    for entry in all_cand_triples:
        new_triples = []
        # triple 'Turn_Me_On_(album) | runtime | 35.1'
        for triple in entry:
            # triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
            new_triple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            new_triple = re.sub(r'_', ' ', new_triple).lower()
            new_triple = re.sub(r'\s+', ' ', new_triple).lower()
            adjusttriple = new_triple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                new_triple = ' | '.join(adjusttriple)
            new_triples.append(new_triple)
        new_cand_list.append(new_triples)

    return allcand_ids, all_text, all_cand_triples, new_cand_list

def getCandsFromTsv(filepath):
    df = pd.read_csv(filepath, sep='\t', header=0)
    print(df.head())
    # df = df[:10]
    # df = df.sort_values(by=['id'])
    # print(df.head())
    # Get the triples for row with id 'Id770'
    # Example of triples: [('Abraham A. Ribicoff', 'born in', 'United States'), ('United States', 'has ethnic group', 'African Americans')]
    # triples_str = df[df['id'] == 'Id770']['triples'].values[0]
    # # Convert the triples string to a list of tuples
    # triples = ast.literal_eval("[" + triples_str + "]")[0]
    allcand_ids = df['id'].values
    all_text = df['lexs'].values
    # from IPython import embed; embed()
    all_cand_triples = []
    for i in range(len(df)):
        new_triples = []
        triples_str = df['triples'].values[i]
        triples = ast.literal_eval("[" + triples_str + "]")[0]
        for triple in triples:
            triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
            new_triples.append(triple_str)
        all_cand_triples.append(new_triples)

    new_cand_list = []
    
    for entry in all_cand_triples:
        new_triples = []
        # triple 'Turn_Me_On_(album) | runtime | 35.1'
        for triple in entry:
            # triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
            new_triple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            new_triple = re.sub(r'_', ' ', new_triple).lower()
            new_triple = re.sub(r'\s+', ' ', new_triple).lower()
            adjusttriple = new_triple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                new_triple = ' | '.join(adjusttriple)
            new_triples.append(new_triple)
        new_cand_list.append(new_triples)

    return allcand_ids, all_text, all_cand_triples, new_cand_list

def getCands(filepath):
    with open(filepath, encoding='utf-8') as fp:
        candssoup = BeautifulSoup(fp, 'lxml')

    candssentries = candssoup.find('benchmark').find('entries').find_all('entry')

    all_cand_triples = []

    for entry in candssentries:
        entrycandtriples = []
        modtriplescand = entry.find('generatedtripleset').find_all('gtriple')
        for modtriple in modtriplescand:
            entrycandtriples.append(modtriple.text)
        all_cand_triples.append(entrycandtriples)

    new_cand_list = []

    for entry in all_cand_triples:
        new_triples = []
        for triple in entry:
            new_triple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            new_triple = re.sub(r'_', ' ', new_triple).lower()
            new_triple = re.sub(r'\s+', ' ', new_triple).lower()
            adjusttriple = new_triple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                new_triple = ' | '.join(adjusttriple)
            new_triples.append(new_triple)
        new_cand_list.append(new_triples)

    return all_cand_triples, new_cand_list

def findSubList(sl,l):
    sll=len(sl)
    #DEBUG: print([i for i,e in enumerate(l) if e==sl[0]])
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

#We are going to try to find matches with the reference, starting with the highest chunk possible (all the words in the reference).
#If we don't find that, we are going to search for all n-grams -1 the number of words in the reference; than -2; than -3; etc.
def nonRefWords(new_ref_list, new_cand_list, foundnum, ngram_length):
    while ngram_length > 0:
        #Get a list of all the ngrams of that size
        ngram_list = list(ngrams(new_cand_list, ngram_length))
        #DEBUG: print("Ngram:", ngram_list)
        for ngram in ngram_list:
            #If we find this ngram (in the same order) in the reference
            #We're getting the start and end index of the ngram in the reference
            find_new_ref = findSubList(list(ngram), new_ref_list)
            if find_new_ref is not None:
                #DEBUG: print("find_new_ref:", find_new_ref)
                #And all the numbers in between
                new_ref_index = list(range(find_new_ref[0], find_new_ref[1] + 1))
                #Change the matched words to FOUNDREF-[FOUNDNUMBER]-[FOUNDINDEX]
                for idx in new_ref_index:
                    new_ref_list[idx] = 'FOUNDREF-' + str(foundnum) + '-' + str(idx)

                #Now find the start and end index of the ngram in the candidate as well
                find_new_cand = findSubList(list(ngram), new_cand_list)
                #And all the indices in between
                new_cand_index = list(range(find_new_cand[0], find_new_cand[1]+1))
                # Change the matched words to FOUNDCAND-[FOUNDNUMBER]-[REFERENCE-FOUNDINDEX]
                for idx, val in enumerate(new_cand_index):
                    new_cand_list[val] = 'FOUNDCAND-' + str(foundnum) + '-' + str(new_ref_index[idx])
                foundnum += 1
                #And try to find new matches again
                nonRefWords(new_ref_list, new_cand_list, foundnum, ngram_length)
        #If no match is found, try to find matches for ngrams 1 smaller
        ngram_length -= 1
    #Return the new lists if all possible ngrams have been searched
    return new_ref_list, new_cand_list

def getRefDict(new_ref_list, new_cand_list, triple_type_ref, triple_type_cand, baseidx):
    try:
        #If some match is found with the reference
        first_found_idx = new_cand_list.index([i for i in new_cand_list if re.findall(r'^FOUNDCAND', i)][0])
        candidate_found = True
    except IndexError:
        candidate_found = False

    if candidate_found:
        unlinked_list = []
        before_list = []
        after_list = []

        #If the first found candidate match is also the first word in the reference
        if new_cand_list[first_found_idx].endswith('-0'):
            #Flag that some words can appear before the first match, and they are linked with the first candidate match
            before_linked = True
            first_cand = re.search(r'^(FOUNDCAND-\d+)-', new_cand_list[first_found_idx]).group(1)
        else:
            before_linked = False

        last_found_idx = None
        after_linked = False
        #If there's more words after the last reference, link those to the last reference as well
        #If the last reference word is linked, but the last candidate word is not, one criterion of linking the last words is met
        if (new_ref_list[-1].startswith('FOUNDREF')) and (not new_cand_list[-1].startswith('FOUNDCAND')):
            #If the last linked reference word is the last linked candidate word, the other criterion is also met.
            last_found = [i for i in new_cand_list if re.findall(r'^FOUNDCAND', i)][-1]
            cand_version = new_ref_list[-1].replace('FOUNDREF', 'FOUNDCAND')
            if last_found == cand_version:
                last_found_idx = new_cand_list.index([i for i in new_cand_list if re.findall(r'^FOUNDCAND', i)][-1])
                after_linked = True
                last_cand = re.search(r'^(FOUNDCAND-\d+)-', last_found).group(1)


        #Ensure that all the not-found blocks are separated by giving them different unlink_numbers
        unlink_number = 1
        for idx, can in enumerate(new_cand_list):
            if not can.startswith('FOUNDCAND'):
                if (idx < first_found_idx) and before_linked:
                    new_cand_list[idx] = first_cand + '-LINKED'
                    before_list.append(first_cand + '-LINKED')
                elif (last_found_idx != None) and (idx > last_found_idx) and after_linked:
                    new_cand_list[idx] = last_cand + '-LINKED'
                    after_list.append(last_cand + '-LINKED')
                else:
                    unlinked_list.append('NOTFOUND-' + str(unlink_number))
            else:
                unlink_number += 1

        total_list = before_list + new_ref_list + after_list + unlinked_list

        ref_start = len(before_list)
        ref_end = (len(before_list) + len(new_ref_list)) - 1

        ref_dict_list = [{'label': triple_type_ref, 'start': baseidx + ref_start, 'end': baseidx + ref_end}]

        total_list2 = [x.replace('FOUNDREF', 'FOUNDCAND') for x in total_list]

        cand_dict_list = []
        current_candidate = ''
        beginidx = ''
        endidx = ''
        collecting = False
        for idx, candidate in enumerate(total_list2):
            if (candidate.startswith('FOUNDCAND')) or (candidate.startswith('NOTFOUND')):
                collecting = True
                curcan = re.search(r'^((.*?)-\d+)', candidate).group(1)
                if curcan != current_candidate:
                    if current_candidate != '':
                        endidx = idx-1
                        cand_dict_list.append({'label': triple_type_cand, 'start': baseidx + beginidx, 'end': baseidx + endidx})
                    current_candidate = curcan
                    beginidx = idx

                if idx == len(total_list2)-1:
                    endidx = idx
                    cand_dict_list.append({'label': triple_type_cand, 'start': baseidx + beginidx, 'end': baseidx + endidx})
            else:
                if collecting:
                    endidx = idx-1
                    cand_dict_list.append({'label': triple_type_cand, 'start': baseidx + beginidx, 'end': baseidx + endidx})

    else:
        if len(new_ref_list) == 0:
            ref_dict_list = []
            cand_dict_list = [{'label': triple_type_cand, 'start': baseidx, 'end': baseidx + (len(new_cand_list) - 1)}]
            total_list = new_cand_list
        elif len(new_cand_list) == 0:
            cand_dict_list = []
            ref_dict_list = [{'label': triple_type_ref, 'start': baseidx, 'end': baseidx + (len(new_ref_list) - 1)}]
            total_list = ref_dict_list
        else:
            total_list = new_ref_list + new_cand_list
            ref_dict_list = [{'label': triple_type_ref, 'start': baseidx, 'end': baseidx + (len(new_ref_list) - 1)}]
            cand_dict_list = [{'label': triple_type_cand, 'start': baseidx + len(new_ref_list), 'end': baseidx + (len(total_list) - 1)}]


    return candidate_found, ref_dict_list, cand_dict_list, total_list

def evaluateRefCand(reference, candidate):
    new_ref = reference.split(' | ')
    new_cand = candidate.split(' | ')
    #DEBUG: print("Ref:", new_ref)
    #DEBUG: print("Cand:", new_cand)

    # Check if triples got split inside a literal value
    # IDEA: Just reconstruct the portion of the list that is a literal that got split.
    # Check for unmatched quotes
    if len(new_ref) > 3:
        if new_ref[0].strip('(').strip(')')[0] == '\"' and new_ref[0].strip('(').strip(')')[-1] != '\"':
            rep_ref = []
            end = 0
            for i in range(1, len(new_ref)):
                new_ref[i] = new_ref[i].strip('\'')
                if new_ref[i].strip('(').strip(')')[-1] == '\"':
                    end = i
            rep_ref.append(", ".join(f'{w.strip("[").strip("]")}' for w in (new_ref[0:end+1])))
            rep_ref.append(new_ref[-2])
            rep_ref.append(new_ref[-1])
            new_ref = rep_ref
        if new_ref[1].strip('(').strip(')')[0] == '\"' and new_ref[1].strip('(').strip(')')[-1] != '\"':
            rep_ref = []
            rep_ref.append(new_ref[0])
            end = 0
            for i in range(1, len(new_ref)):
                new_ref[i] = new_ref[i].strip('\'')
                if new_ref[i].strip('(').strip(')')[-1] == '\"':
                    end = i
            rep_ref.append(", ".join(f'{w.strip("[").strip("]")}' for w in (new_ref[1:end+1])))
            rep_ref.append(new_ref[-1])
            new_ref = rep_ref
        if new_ref[2].strip('(').strip(')')[0] == '\"' and new_ref[2].strip('(').strip(')')[-1] != '\"':
            rep_ref = []
            rep_ref.append(new_ref[0])
            rep_ref.append(new_ref[1])
            end = 0
            for i in range(2, len(new_ref)):
                new_ref[i] = new_ref[i].strip('\'')
                if new_ref[i].strip('(').strip(')')[-1] == '\"':
                    end = i
            rep_ref.append(", ".join(f'{w.strip("[").strip("]")}' for w in (new_ref[2:end+1])))
            new_ref = rep_ref
    
    if len(new_cand) > 3:
        if new_cand[0].strip('(').strip(')')[0] == '\"' and new_cand[0].strip('(').strip(')')[-1] != '\"':
            rep_cand = []
            end = 0
            for i in range(1, len(new_cand)):
                new_cand[i] = new_cand[i].strip('\'')
                if new_cand[i][-1].strip('(').strip(')') == '\"':
                    end = i
            rep_cand.append(", ".join(f'{w.strip("[").strip("]")}' for w in (new_cand[0:end+1])))
            rep_cand.append(new_cand[-2])
            rep_cand.append(new_cand[-1])
            new_cand = rep_cand
        if new_cand[1].strip('(').strip(')')[0] == '\"' and new_cand[1].strip('(').strip(')')[-1] != '\"':
            rep_cand = []
            rep_cand.append(new_cand[0])
            end = 0
            for i in range(1, len(new_cand)):
                new_cand[i] = new_cand[i].strip('\'')
                if new_cand[i][-1].strip('(').strip(')') == '\"':
                    end = i
            rep_cand.append(", ".join(f'{w.strip("[").strip("]")}' for w in (new_cand[1:end+1])))
            rep_cand.append(new_cand[-1])
            new_cand = rep_cand
        if new_cand[2].strip('(').strip(')')[0] == '\"' and new_cand[2].strip('(').strip(')')[-1] != '\"':
            rep_cand = []
            rep_cand.append(new_cand[0])
            rep_cand.append(new_cand[1])
            end = 0
            for i in range(1, len(new_cand)):
                new_cand[i] = new_cand[i].strip('\'')
                if new_cand[i][-1].strip('(').strip(')') == '\"':
                    end = i
            rep_cand.append(", ".join(f'{w.strip("[").strip("]")}' for w in (new_cand[2:end+1])))
            new_cand = rep_cand



    # Make sure that reference or candidate aren't '' values originally.
    if (len(new_ref) > 1) and (len(new_cand) > 1):
        index_triple = new_ref
    elif (len(new_ref) == 1) :
        index_triple = new_cand
        new_ref = ['', '', '']
    else:
        index_triple = new_ref
        new_cand = ['', '', '']

    subject_ref_list = None
    subject_cand_list = None
    subject_total_list = None
    predicate_ref_list = None
    predicate_cand_list = None
    predicate_total_list = None
    object_ref_list = None
    object_cand_list = None
    object_total_list = None
    subject_found = False
    predicate_found = False
    object_found = False

    for idx, attrib in enumerate(index_triple):
        #Let's go over each attribute of the triple one by one
        try:
            refsub = new_ref[idx]
            candsub = new_cand[idx]
        except IndexError as i:
            print(i)
            print("idx:",idx)
            print("refsub:",refsub)
            print("candsub:",candsub)
            print("reflist:",new_ref,len(new_ref))
            print("candlist:",new_cand,len(new_cand))
            exit(0)

        ref_list = nltk.word_tokenize(refsub)
        cand_list = nltk.word_tokenize(candsub)

        ref_list = [x.lower() for x in ref_list if re.search(r'^[' + re.escape(string.punctuation) + r']+$', x) == None]
        cand_list = [x.lower() for x in cand_list if re.search(r'^[' + re.escape(string.punctuation) + r']+$', x) == None]
        #DEBUG: print("Ref List:", ref_list)
        #DEBUG: print("Cand List:", cand_list)

        new_ref_list = ref_list.copy()
        new_cand_list = cand_list.copy()
        # Start with an ngram the full number of words in the reference
        ngram_length = len(new_cand_list)
        new_ref_list, new_cand_list = nonRefWords(new_ref_list, new_cand_list, 1, ngram_length)
        if idx == 0:
            candidate_found, ref_dict_list, cand_dict_list, total_list = getRefDict(new_ref_list, new_cand_list, 'SUB', 'SUB', 0)
            subject_found = candidate_found
            subject_ref_list = ref_dict_list.copy()
            subject_cand_list = cand_dict_list.copy()
            subject_total_list = total_list.copy()
        elif idx == 1:
            candidate_found, ref_dict_list, cand_dict_list, total_list = getRefDict(new_ref_list, new_cand_list, 'PRED', 'PRED', len(subject_total_list))
            predicate_found = candidate_found
            predicate_ref_list = ref_dict_list.copy()
            predicate_cand_list = cand_dict_list.copy()
            predicate_total_list = total_list.copy()
        else:
            candidate_found, ref_dict_list, cand_dict_list, total_list = getRefDict(new_ref_list, new_cand_list, 'OBJ', 'OBJ', len(subject_total_list) + len(predicate_total_list))
            object_found = candidate_found
            object_ref_list = ref_dict_list.copy()
            object_cand_list = cand_dict_list.copy()
            object_total_list = total_list.copy()

    switch_match_found = False
    #If no matches were found for two or more attributes, we are going to try and compare different attributes to each other.
    #First let's try to match the candidate subject and reference object (and vice versa)
    if not subject_found and not object_found:
        refsub = new_ref[0]
        candsub = new_cand[2]

        ref_list = nltk.word_tokenize(refsub)
        cand_list = nltk.word_tokenize(candsub)

        ref_list = [x.lower() for x in ref_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        cand_list = [x.lower() for x in cand_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        new_ref_list = ref_list.copy()
        new_cand_list = cand_list.copy()
        # Start with an ngram the full number of words in the candidate
        ngram_length = len(new_cand_list)
        new_ref_list, new_cand_list = nonRefWords(new_ref_list, new_cand_list, 1, ngram_length)

        candidate_found, ref_dict_list, cand_dict_list, total_list = getRefDict(new_ref_list, new_cand_list, 'SUB', 'OBJ', 0)

        refsub = new_ref[2]
        candsub = new_cand[0]

        ref_list = nltk.word_tokenize(refsub)
        cand_list = nltk.word_tokenize(candsub)

        ref_list = [x.lower() for x in ref_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        cand_list = [x.lower() for x in cand_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        new_ref_list = ref_list.copy()
        new_cand_list = cand_list.copy()
        # Start with an ngram the full number of words in the candidate
        ngram_length = len(new_cand_list)
        new_ref_list, new_cand_list = nonRefWords(new_ref_list, new_cand_list, 1, ngram_length)
        candidate_found2, ref_dict_list2, cand_dict_list2, total_list2 = getRefDict(new_ref_list, new_cand_list, 'OBJ', 'SUB', len(total_list) + len(predicate_total_list))

        # subject_found is based in reference 
        if candidate_found or candidate_found2:
            subject_found = candidate_found
            subject_ref_list = ref_dict_list.copy()
            subject_cand_list = cand_dict_list.copy()
            subject_total_list = total_list.copy()
            object_found = candidate_found2
            object_ref_list = ref_dict_list2.copy()
            object_cand_list = cand_dict_list2.copy()
            object_total_list = total_list2.copy()

            # get the new mapping of the predicate
            refpred = new_ref[1]
            candpred = new_cand[1]

            ref_list = nltk.word_tokenize(refpred)
            cand_list = nltk.word_tokenize(candpred)

            ref_list = [x.lower() for x in ref_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
            cand_list = [x.lower() for x in cand_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

            new_ref_list = ref_list.copy()
            new_cand_list = cand_list.copy()
            # Start with an ngram the full number of words in the candidate
            ngram_length = len(new_cand_list)
            new_ref_list, new_cand_list = nonRefWords(new_ref_list, new_cand_list, 1, ngram_length)

            # debugging wrong code here nothing to do with predicate
            candidate_found, ref_dict_list, cand_dict_list, total_list = getRefDict(new_ref_list, new_cand_list, 'PRED', 'PRED', len(subject_total_list))
            predicate_found = candidate_found
            predicate_ref_list = ref_dict_list.copy()
            predicate_cand_list = cand_dict_list.copy()
            predicate_total_list = total_list.copy()

            switch_match_found = True
        else:
            switch_match_found = False

    # Then, let's try to switch subject and predicate
    if (not subject_found and not predicate_found) and not switch_match_found:
        refsub = new_ref[0]
        candsub = new_cand[1]

        ref_list = nltk.word_tokenize(refsub)
        cand_list = nltk.word_tokenize(candsub)

        ref_list = [x.lower() for x in ref_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        cand_list = [x.lower() for x in cand_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        new_ref_list = ref_list.copy()
        new_cand_list = cand_list.copy()
        # Start with an ngram the full number of words in the candidate
        ngram_length = len(new_cand_list)
        new_ref_list, new_cand_list = nonRefWords(new_ref_list, new_cand_list, 1, ngram_length)

        candidate_found, ref_dict_list, cand_dict_list, total_list = getRefDict(new_ref_list, new_cand_list, 'SUB', 'PRED', 0)

        refsub = new_ref[1]
        candsub = new_cand[0]

        ref_list = nltk.word_tokenize(refsub)
        cand_list = nltk.word_tokenize(candsub)

        ref_list = [x.lower() for x in ref_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        cand_list = [x.lower() for x in cand_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        new_ref_list = ref_list.copy()
        new_cand_list = cand_list.copy()
        # Start with an ngram the full number of words in the candidate
        ngram_length = len(new_cand_list)
        new_ref_list, new_cand_list = nonRefWords(new_ref_list, new_cand_list, 1, ngram_length)

        candidate_found2, ref_dict_list2, cand_dict_list2, total_list2 = getRefDict(new_ref_list, new_cand_list, 'PRED', 'SUB', len(total_list))

        if candidate_found or candidate_found2:
            subject_found = candidate_found
            subject_ref_list = ref_dict_list.copy()
            subject_cand_list = cand_dict_list.copy()
            subject_total_list = total_list.copy()
            predicate_found = candidate_found2
            predicate_ref_list = ref_dict_list2.copy()
            predicate_cand_list = cand_dict_list2.copy()
            predicate_total_list = total_list2.copy()
            switch_match_found = True
        else:
            switch_match_found = False

    # Finally, let's try to switch predicate and object
    if (not predicate_found and not object_found) and not switch_match_found:
        refsub = new_ref[1]
        candsub = new_cand[2]

        ref_list = nltk.word_tokenize(refsub)
        cand_list = nltk.word_tokenize(candsub)

        ref_list = [x.lower() for x in ref_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        cand_list = [x.lower() for x in cand_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        new_ref_list = ref_list.copy()
        new_cand_list = cand_list.copy()
        # Start with an ngram the full number of words in the candidate
        ngram_length = len(new_cand_list)
        new_ref_list, new_cand_list = nonRefWords(new_ref_list, new_cand_list, 1, ngram_length)

        candidate_found, ref_dict_list, cand_dict_list, total_list = getRefDict(new_ref_list, new_cand_list, 'PRED', 'OBJ', len(subject_total_list))

        refsub = new_ref[2]
        candsub = new_cand[1]

        ref_list = nltk.word_tokenize(refsub)
        cand_list = nltk.word_tokenize(candsub)

        ref_list = [x.lower() for x in ref_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        cand_list = [x.lower() for x in cand_list if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        new_ref_list = ref_list.copy()
        new_cand_list = cand_list.copy()
        # Start with an ngram the full number of words in the candidate
        ngram_length = len(new_cand_list)
        new_ref_list, new_cand_list = nonRefWords(new_ref_list, new_cand_list, 1, ngram_length)

        candidate_found2, ref_dict_list2, cand_dict_list2, total_list2 = getRefDict(new_ref_list, new_cand_list, 'OBJ', 'PRED', len(subject_total_list) + len(total_list))

        if candidate_found or candidate_found2:
            predicate_found = candidate_found
            predicate_ref_list = ref_dict_list.copy()
            predicate_cand_list = cand_dict_list.copy()
            predicate_total_list = total_list.copy()
            object_found = candidate_found2
            object_ref_list = ref_dict_list2.copy()
            object_cand_list = cand_dict_list2.copy()
            object_total_list = total_list2.copy()
            switch_match_found = True
        else:
            switch_match_found = False

    all_ref_dict = subject_ref_list + predicate_ref_list + object_ref_list
    all_cand_dict = subject_cand_list + predicate_cand_list + object_cand_list
    all_total_list = subject_total_list + predicate_total_list + object_total_list

    evaluator = Evaluator([all_ref_dict], [all_cand_dict], tags=['SUB', 'PRED', 'OBJ'])

    # Returns overall metrics and metrics for each tag

    results, results_per_tag = evaluator.evaluate()

    return results, results_per_tag

def calculateAllScores(new_ref_list, new_cand_list):
    #DEBUG: print(new_ref_list)
    #DEBUG: print(new_cand_list)
    total_sem_eval_list = []
    total_sem_eval_list_per_tag = []

    for idx, candidate in enumerate(new_cand_list):
        #print('evaluating candidate ' + str(idx) + ' of ' + str(len(new_cand_list)))
        # Ensure list lengths are equal, pad with empty strings.
        if len(new_cand_list[idx]) != len(new_ref_list[idx]):
            difference_between = abs(len(new_cand_list[idx]) - len(new_ref_list[idx]))
            difference_list = [''] * difference_between
            if len(new_cand_list[idx]) < len(new_ref_list[idx]):
                new_cand_list[idx] = new_cand_list[idx] + difference_list
            else:
                new_ref_list[idx] = new_ref_list[idx] + difference_list

    # Evaluate every triple against each reference triple
    for idx, candidate in enumerate(new_cand_list):
        candidate_sem_eval = []
        candidate_sem_eval_per_tag = []
        for triple in candidate:
            triple_sem_eval = []
            triple_sem_eval_per_tag = []
            for reference in new_ref_list[idx]:
                results, results_per_tag = evaluateRefCand(reference, triple)
                triple_sem_eval.append(results)
                triple_sem_eval_per_tag.append(results_per_tag)

            candidate_sem_eval.append(triple_sem_eval)
            candidate_sem_eval_per_tag.append(triple_sem_eval_per_tag)

        total_sem_eval_list.append(candidate_sem_eval)
        total_sem_eval_list_per_tag.append(candidate_sem_eval_per_tag)

    return total_sem_eval_list, total_sem_eval_list_per_tag

def sumAllCombination(selected_sem_eval_list):
    # import IPython; IPython.embed()
    all_dict = {}
    # all_dict.update({'Total_scores': {}})
    for key in selected_sem_eval_list[0].keys():
        ent_type_correct = sum([x[key]['correct'] for x in selected_sem_eval_list])
        ent_type_incorrect = sum([x[key]['incorrect'] for x in selected_sem_eval_list])
        ent_type_partial = sum([x[key]['partial'] for x in selected_sem_eval_list])
        ent_type_missed = sum([x[key]['missed'] for x in selected_sem_eval_list])
        ent_type_spurious = sum([x[key]['spurious'] for x in selected_sem_eval_list])
        ent_type_possible = sum([x[key]['possible'] for x in selected_sem_eval_list])
        ent_type_actual = sum([x[key]['actual'] for x in selected_sem_eval_list])
        ent_type_precision = statistics.mean([x[key]['precision'] for x in selected_sem_eval_list])
        ent_type_recall = statistics.mean([x[key]['recall'] for x in selected_sem_eval_list])
        ent_type_f1 = statistics.mean([x[key]['f1'] for x in selected_sem_eval_list])

        ent_type_dict = {key: {'Correct': ent_type_correct, 'Incorrect': ent_type_incorrect, 'Partial': ent_type_partial, 'Missed': ent_type_missed,
                                    'Spurious': ent_type_spurious, 'Possible': ent_type_possible, 'Actual': ent_type_actual, 'Precision': ent_type_precision,
                                    'Recall': ent_type_recall, 'F1': ent_type_f1}}
        all_dict.update(ent_type_dict)
    return all_dict

def calculateSystemScore(total_sem_eval_list, total_sem_eval_list_per_tag, new_ref_list, new_cand_list):
    selected_sem_eval_list = []
    selected_sem_eval_list_per_tag = []
    triple_score = []
    combination_selected = []
    triple_score_sum = []

    # Get all the permutations of the number of scores given per candidate, so if there's 4 candidates, but 3 references, this part ensures that one of
    # The four will not be scored
    for idx, candidate in enumerate(new_cand_list):
        #print('calculating system score for candidate ' + str(idx) + ' of ' + str(len(new_cand_list)))
        # if len(new_cand_list[idx]) > len(new_ref_list[idx]):
        #     # Get all permutations
        #     choosecands = list(itertools.permutations([x[0] for x in enumerate(total_sem_eval_list[idx])], len(total_sem_eval_list[idx][0])))
        #     # The permutations in different orders are not necessary: we only need one order without the number of candidates we're looking at
        #     choosecands = set([tuple(sorted(i)) for i in choosecands])  # Sort inner list and then use set
        #     choosecands = list(map(list, choosecands))  # Converting back to list
        # else:
        #     # Otherwise, we're just going to score all candidates
        #     choosecands = [list(range(len(new_cand_list[idx])))]

        # # Get all permutations in which the scores can be combined
        # if len(new_cand_list[idx]) > len(new_ref_list[idx]):
        #     choosescore = list(itertools.permutations([x[0] for x in enumerate(total_sem_eval_list[idx][0])], len(new_ref_list[idx])))
        #     choosescore = [list(x) for x in choosescore]
        # else:
        #     choosescore = list(itertools.permutations([x[0] for x in enumerate(total_sem_eval_list[idx][0])], len(new_cand_list[idx])))
        #     choosescore = [list(x) for x in choosescore]

        # # Get all possible combinations between the candidates and the scores
        # combilist = list(itertools.product(choosecands, choosescore))

        total_dict = {'totalscore': 0}

        # for combination in combilist:
        #     combi_score = 0
        #     # Take the combination between the candidate and the score
        #     zipcombi = list(zip(combination[0], combination[1]))
        #     collected_sem_eval = []
        #     collected_sem_eval_per_tag = []

        #     for zc in zipcombi:
        #         collected_scores = total_sem_eval_list[idx][zc[0]][zc[1]]
        #         f1_score = statistics.mean([collected_scores['ent_type']['f1'], collected_scores['partial']['f1'], collected_scores['strict']['f1'], collected_scores['exact']['f1']])
        #         combi_score += f1_score

        #         collected_sem_eval.append(collected_scores)
        #         collected_sem_eval_per_tag.append(total_sem_eval_list_per_tag[idx][zc[0]][zc[1]])


        #     # If the combination is the highest score thus far, or the first score, make it the total_dict
        #     if (combi_score > total_dict['totalscore']) or (len(total_dict) == 1):
        #         total_dict = {'totalscore': combi_score, 'combination': combination, 'sem_eval_list': collected_sem_eval,
        #                      'sem_eval_per_tag_list': collected_sem_eval_per_tag}
        # triple_score.append(total_dict['sem_eval_list'])
        # combination_selected.append(total_dict['combination'])
        # ent_type_dict = sumAllCombination(total_dict['sem_eval_list'])
        # triple_score_sum.append(ent_type_dict)
        # selected_sem_eval_list = selected_sem_eval_list + total_dict['sem_eval_list']
        # selected_sem_eval_list_per_tag = selected_sem_eval_list_per_tag + total_dict['sem_eval_per_tag_list']
        collected_sem_eval = []
        collected_sem_eval_per_tag = []
        collected_combinations = []
        for ind_cand in range(len(new_cand_list[idx])):
            combi_score = 0
            # Take the combination between the candidate and the score
            f1_scores = []
            combination = []
            
            for ind_ref in range(len(new_ref_list[idx])):
                collected_scores = total_sem_eval_list[idx][ind_cand][ind_ref]
                f1_score = statistics.mean([collected_scores['ent_type']['f1'], collected_scores['partial']['f1'], collected_scores['strict']['f1'], collected_scores['exact']['f1']])
                f1_scores.append(f1_score)
                combination.append([ind_cand, ind_ref])

            # If the combination is the highest score thus far, or the first score, make it the total_dict
            index_max = np.argmax(f1_scores)
            selected_combination = combination[index_max]
            collected_combinations.append(selected_combination)
            collected_sem_eval.append(total_sem_eval_list[idx][selected_combination[0]][selected_combination[1]])
            collected_sem_eval_per_tag.append(total_sem_eval_list_per_tag[idx][selected_combination[0]][selected_combination[1]])
            combi_score = f1_score
            total_dict = {'totalscore': combi_score, 'combination': collected_combinations, 'sem_eval_list': collected_sem_eval,
                        'sem_eval_per_tag_list': collected_sem_eval_per_tag}
        triple_score.append(total_dict['sem_eval_list'])
        combination_selected.append(total_dict['combination'])
        ent_type_dict = sumAllCombination(total_dict['sem_eval_list'])
        triple_score_sum.append(ent_type_dict)
        selected_sem_eval_list = selected_sem_eval_list + total_dict['sem_eval_list']
        selected_sem_eval_list_per_tag = selected_sem_eval_list_per_tag + total_dict['sem_eval_per_tag_list']

    all_dict = {}
    all_dict.update({'Total_scores': {}})

    ent_type_correct = sum([x['ent_type']['correct'] for x in selected_sem_eval_list])
    ent_type_incorrect = sum([x['ent_type']['incorrect'] for x in selected_sem_eval_list])
    ent_type_partial = sum([x['ent_type']['partial'] for x in selected_sem_eval_list])
    ent_type_missed = sum([x['ent_type']['missed'] for x in selected_sem_eval_list])
    ent_type_spurious = sum([x['ent_type']['spurious'] for x in selected_sem_eval_list])
    ent_type_possible = sum([x['ent_type']['possible'] for x in selected_sem_eval_list])
    ent_type_actual = sum([x['ent_type']['actual'] for x in selected_sem_eval_list])
    ent_type_precision = statistics.mean([x['ent_type']['precision'] for x in selected_sem_eval_list])
    ent_type_recall = statistics.mean([x['ent_type']['recall'] for x in selected_sem_eval_list])
    ent_type_f1 = statistics.mean([x['ent_type']['f1'] for x in selected_sem_eval_list])

    ent_type_dict = {'Ent_type': {'Correct': ent_type_correct, 'Incorrect': ent_type_incorrect, 'Partial': ent_type_partial, 'Missed': ent_type_missed,
                                'Spurious': ent_type_spurious, 'Possible': ent_type_possible, 'Actual': ent_type_actual, 'Precision': ent_type_precision,
                                'Recall': ent_type_recall, 'F1': ent_type_f1}}

    all_dict['Total_scores'].update(ent_type_dict)

    partial_correct = sum([x['partial']['correct'] for x in selected_sem_eval_list])
    partial_incorrect = sum([x['partial']['incorrect'] for x in selected_sem_eval_list])
    partial_partial = sum([x['partial']['partial'] for x in selected_sem_eval_list])
    partial_missed = sum([x['partial']['missed'] for x in selected_sem_eval_list])
    partial_spurious = sum([x['partial']['spurious'] for x in selected_sem_eval_list])
    partial_possible = sum([x['partial']['possible'] for x in selected_sem_eval_list])
    partial_actual = sum([x['partial']['actual'] for x in selected_sem_eval_list])
    partial_precision = statistics.mean([x['partial']['precision'] for x in selected_sem_eval_list])
    partial_recall = statistics.mean([x['partial']['recall'] for x in selected_sem_eval_list])
    partial_f1 = statistics.mean([x['partial']['f1'] for x in selected_sem_eval_list])

    partial_dict = {'Partial': {'Correct': partial_correct, 'Incorrect': partial_incorrect, 'Partial': partial_partial, 'Missed': partial_missed,
                                'Spurious': partial_spurious, 'Possible': partial_possible, 'Actual': partial_actual, 'Precision': partial_precision,
                                'Recall': partial_recall, 'F1': partial_f1}}
    all_dict['Total_scores'].update(partial_dict)

    strict_correct = sum([x['strict']['correct'] for x in selected_sem_eval_list])
    strict_incorrect = sum([x['strict']['incorrect'] for x in selected_sem_eval_list])
    strict_partial = sum([x['strict']['partial'] for x in selected_sem_eval_list])
    strict_missed = sum([x['strict']['missed'] for x in selected_sem_eval_list])
    strict_spurious = sum([x['strict']['spurious'] for x in selected_sem_eval_list])
    strict_possible = sum([x['strict']['possible'] for x in selected_sem_eval_list])
    strict_actual = sum([x['strict']['actual'] for x in selected_sem_eval_list])
    strict_precision = statistics.mean([x['strict']['precision'] for x in selected_sem_eval_list])
    strict_recall = statistics.mean([x['strict']['recall'] for x in selected_sem_eval_list])
    strict_f1 = statistics.mean([x['strict']['f1'] for x in selected_sem_eval_list])

    strict_dict = {'Strict': {'Correct': strict_correct, 'Incorrect': strict_incorrect, 'Partial': strict_partial, 'Missed': strict_missed,
                                'Spurious': strict_spurious, 'Possible': strict_possible, 'Actual': strict_actual, 'Precision': strict_precision,
                                'Recall': strict_recall, 'F1': strict_f1}}
    all_dict['Total_scores'].update(strict_dict)

    exact_correct = sum([x['exact']['correct'] for x in selected_sem_eval_list])
    exact_incorrect = sum([x['exact']['incorrect'] for x in selected_sem_eval_list])
    exact_partial = sum([x['exact']['partial'] for x in selected_sem_eval_list])
    exact_missed = sum([x['exact']['missed'] for x in selected_sem_eval_list])
    exact_spurious = sum([x['exact']['spurious'] for x in selected_sem_eval_list])
    exact_possible = sum([x['exact']['possible'] for x in selected_sem_eval_list])
    exact_actual = sum([x['exact']['actual'] for x in selected_sem_eval_list])
    exact_precision = statistics.mean([x['exact']['precision'] for x in selected_sem_eval_list])
    exact_recall = statistics.mean([x['exact']['recall'] for x in selected_sem_eval_list])
    exact_f1 = statistics.mean([x['exact']['f1'] for x in selected_sem_eval_list])

    exact_dict = {'Exact': {'Correct': exact_correct, 'Incorrect': exact_incorrect, 'Partial': exact_partial, 'Missed': exact_missed,
                                'Spurious': exact_spurious, 'Possible': exact_possible, 'Actual': exact_actual, 'Precision': exact_precision,
                                'Recall': exact_recall, 'F1': exact_f1}}
    all_dict['Total_scores'].update(exact_dict)

    all_dict.update({'Scores_per_tag': {}})

    all_dict['Scores_per_tag'].update({'Subjects': {}})

    sub_ent_type_correct = sum([x['SUB']['ent_type']['correct'] for x in selected_sem_eval_list_per_tag])
    sub_ent_type_incorrect = sum([x['SUB']['ent_type']['incorrect'] for x in selected_sem_eval_list_per_tag])
    sub_ent_type_partial = sum([x['SUB']['ent_type']['partial'] for x in selected_sem_eval_list_per_tag])
    sub_ent_type_missed = sum([x['SUB']['ent_type']['missed'] for x in selected_sem_eval_list_per_tag])
    sub_ent_type_spurious = sum([x['SUB']['ent_type']['spurious'] for x in selected_sem_eval_list_per_tag])
    sub_ent_type_possible = sum([x['SUB']['ent_type']['possible'] for x in selected_sem_eval_list_per_tag])
    sub_ent_type_actual = sum([x['SUB']['ent_type']['actual'] for x in selected_sem_eval_list_per_tag])
    sub_ent_type_precision = statistics.mean([x['SUB']['ent_type']['precision'] for x in selected_sem_eval_list_per_tag])
    sub_ent_type_recall = statistics.mean([x['SUB']['ent_type']['recall'] for x in selected_sem_eval_list_per_tag])
    sub_ent_type_f1 = statistics.mean([x['SUB']['ent_type']['f1'] for x in selected_sem_eval_list_per_tag])

    sub_ent_type_dict = {'Ent_type': {'Correct': sub_ent_type_correct, 'Incorrect': sub_ent_type_incorrect, 'Partial': sub_ent_type_partial, 'Missed': sub_ent_type_missed,
                           'Spurious': sub_ent_type_spurious, 'Possible': sub_ent_type_possible, 'Actual': sub_ent_type_actual, 'Precision': sub_ent_type_precision,
                           'Recall': sub_ent_type_recall, 'F1': sub_ent_type_f1}}
    all_dict['Scores_per_tag']['Subjects'].update(sub_ent_type_dict)

    sub_partial_correct = sum([x['SUB']['partial']['correct'] for x in selected_sem_eval_list_per_tag])
    sub_partial_incorrect = sum([x['SUB']['partial']['incorrect'] for x in selected_sem_eval_list_per_tag])
    sub_partial_partial = sum([x['SUB']['partial']['partial'] for x in selected_sem_eval_list_per_tag])
    sub_partial_missed = sum([x['SUB']['partial']['missed'] for x in selected_sem_eval_list_per_tag])
    sub_partial_spurious = sum([x['SUB']['partial']['spurious'] for x in selected_sem_eval_list_per_tag])
    sub_partial_possible = sum([x['SUB']['partial']['possible'] for x in selected_sem_eval_list_per_tag])
    sub_partial_actual = sum([x['SUB']['partial']['actual'] for x in selected_sem_eval_list_per_tag])
    sub_partial_precision = statistics.mean([x['SUB']['partial']['precision'] for x in selected_sem_eval_list_per_tag])
    sub_partial_recall = statistics.mean([x['SUB']['partial']['recall'] for x in selected_sem_eval_list_per_tag])
    sub_partial_f1 = statistics.mean([x['SUB']['partial']['f1'] for x in selected_sem_eval_list_per_tag])

    sub_partial_dict = {'Partial': {'Correct': sub_partial_correct, 'Incorrect': sub_partial_incorrect, 'Partial': sub_partial_partial, 'Missed': sub_partial_missed,
                           'Spurious': sub_partial_spurious, 'Possible': sub_partial_possible, 'Actual': sub_partial_actual, 'Precision': sub_partial_precision,
                           'Recall': sub_partial_recall, 'F1': sub_partial_f1}}
    all_dict['Scores_per_tag']['Subjects'].update(sub_partial_dict)

    sub_strict_correct = sum([x['SUB']['strict']['correct'] for x in selected_sem_eval_list_per_tag])
    sub_strict_incorrect = sum([x['SUB']['strict']['incorrect'] for x in selected_sem_eval_list_per_tag])
    sub_strict_partial = sum([x['SUB']['strict']['partial'] for x in selected_sem_eval_list_per_tag])
    sub_strict_missed = sum([x['SUB']['strict']['missed'] for x in selected_sem_eval_list_per_tag])
    sub_strict_spurious = sum([x['SUB']['strict']['spurious'] for x in selected_sem_eval_list_per_tag])
    sub_strict_possible = sum([x['SUB']['strict']['possible'] for x in selected_sem_eval_list_per_tag])
    sub_strict_actual = sum([x['SUB']['strict']['actual'] for x in selected_sem_eval_list_per_tag])
    sub_strict_precision = statistics.mean([x['SUB']['strict']['precision'] for x in selected_sem_eval_list_per_tag])
    sub_strict_recall = statistics.mean([x['SUB']['strict']['recall'] for x in selected_sem_eval_list_per_tag])
    sub_strict_f1 = statistics.mean([x['SUB']['strict']['f1'] for x in selected_sem_eval_list_per_tag])

    sub_strict_dict = {'Strict': {'Correct': sub_strict_correct, 'Incorrect': sub_strict_incorrect, 'Partial': sub_strict_partial, 'Missed': sub_strict_missed,
                           'Spurious': sub_strict_spurious, 'Possible': sub_strict_possible, 'Actual': sub_strict_actual, 'Precision': sub_strict_precision,
                           'Recall': sub_strict_recall, 'F1': sub_strict_f1}}
    all_dict['Scores_per_tag']['Subjects'].update(sub_strict_dict)

    sub_exact_correct = sum([x['SUB']['exact']['correct'] for x in selected_sem_eval_list_per_tag])
    sub_exact_incorrect = sum([x['SUB']['exact']['incorrect'] for x in selected_sem_eval_list_per_tag])
    sub_exact_partial = sum([x['SUB']['exact']['partial'] for x in selected_sem_eval_list_per_tag])
    sub_exact_missed = sum([x['SUB']['exact']['missed'] for x in selected_sem_eval_list_per_tag])
    sub_exact_spurious = sum([x['SUB']['exact']['spurious'] for x in selected_sem_eval_list_per_tag])
    sub_exact_possible = sum([x['SUB']['exact']['possible'] for x in selected_sem_eval_list_per_tag])
    sub_exact_actual = sum([x['SUB']['exact']['actual'] for x in selected_sem_eval_list_per_tag])
    sub_exact_precision = statistics.mean([x['SUB']['exact']['precision'] for x in selected_sem_eval_list_per_tag])
    sub_exact_reacall = statistics.mean([x['SUB']['exact']['recall'] for x in selected_sem_eval_list_per_tag])
    sub_exact_f1 = statistics.mean([x['SUB']['exact']['f1'] for x in selected_sem_eval_list_per_tag])

    sub_exact_dict = {'Exact': {'Correct': sub_exact_correct, 'Incorrect': sub_exact_incorrect, 'Partial': sub_exact_partial, 'Missed': sub_exact_missed,
                                'Spurious': sub_exact_spurious, 'Possible': sub_exact_possible, 'Actual': sub_exact_actual,
                                'Precision': sub_exact_precision,
                                'Recall': sub_exact_reacall, 'F1': sub_exact_f1}}
    all_dict['Scores_per_tag']['Subjects'].update(sub_exact_dict)

    all_dict['Scores_per_tag'].update({'Predicates': {}})

    pred_ent_type_correct = sum([x['PRED']['ent_type']['correct'] for x in selected_sem_eval_list_per_tag])
    pred_ent_type_incorrect = sum([x['PRED']['ent_type']['incorrect'] for x in selected_sem_eval_list_per_tag])
    pred_ent_type_partial = sum([x['PRED']['ent_type']['partial'] for x in selected_sem_eval_list_per_tag])
    pred_ent_type_missed = sum([x['PRED']['ent_type']['missed'] for x in selected_sem_eval_list_per_tag])
    pred_ent_type_spurious = sum([x['PRED']['ent_type']['spurious'] for x in selected_sem_eval_list_per_tag])
    pred_ent_type_possible = sum([x['PRED']['ent_type']['possible'] for x in selected_sem_eval_list_per_tag])
    pred_ent_type_actual = sum([x['PRED']['ent_type']['actual'] for x in selected_sem_eval_list_per_tag])
    pred_ent_type_precision = statistics.mean([x['PRED']['ent_type']['precision'] for x in selected_sem_eval_list_per_tag])
    pred_ent_type_recall = statistics.mean([x['PRED']['ent_type']['recall'] for x in selected_sem_eval_list_per_tag])
    pred_ent_type_f1 = statistics.mean([x['PRED']['ent_type']['f1'] for x in selected_sem_eval_list_per_tag])

    pred_ent_type_dict = {
        'Ent_type': {'Correct': pred_ent_type_correct, 'Incorrect': pred_ent_type_incorrect, 'Partial': pred_ent_type_partial, 'Missed': pred_ent_type_missed,
                     'Spurious': pred_ent_type_spurious, 'Possible': pred_ent_type_possible, 'Actual': pred_ent_type_actual, 'Precision': pred_ent_type_precision,
                     'Recall': pred_ent_type_recall, 'F1': pred_ent_type_f1}}
    all_dict['Scores_per_tag']['Predicates'].update(pred_ent_type_dict)

    pred_partial_correct = sum([x['PRED']['partial']['correct'] for x in selected_sem_eval_list_per_tag])
    pred_partial_incorrect = sum([x['PRED']['partial']['incorrect'] for x in selected_sem_eval_list_per_tag])
    pred_partial_partial = sum([x['PRED']['partial']['partial'] for x in selected_sem_eval_list_per_tag])
    pred_partial_missed = sum([x['PRED']['partial']['missed'] for x in selected_sem_eval_list_per_tag])
    pred_partial_spurious = sum([x['PRED']['partial']['spurious'] for x in selected_sem_eval_list_per_tag])
    pred_partial_possible = sum([x['PRED']['partial']['possible'] for x in selected_sem_eval_list_per_tag])
    pred_partial_actual = sum([x['PRED']['partial']['actual'] for x in selected_sem_eval_list_per_tag])
    pred_partial_precision = statistics.mean([x['PRED']['partial']['precision'] for x in selected_sem_eval_list_per_tag])
    pred_partial_recall = statistics.mean([x['PRED']['partial']['recall'] for x in selected_sem_eval_list_per_tag])
    pred_partial_f1 = statistics.mean([x['PRED']['partial']['f1'] for x in selected_sem_eval_list_per_tag])

    pred_partial_dict = {
        'Partial': {'Correct': pred_partial_correct, 'Incorrect': pred_partial_incorrect, 'Partial': pred_partial_partial, 'Missed': pred_partial_missed,
                    'Spurious': pred_partial_spurious, 'Possible': pred_partial_possible, 'Actual': pred_partial_actual, 'Precision': pred_partial_precision,
                    'Recall': pred_partial_recall, 'F1': pred_partial_f1}}
    all_dict['Scores_per_tag']['Predicates'].update(pred_partial_dict)

    pred_strict_correct = sum([x['PRED']['strict']['correct'] for x in selected_sem_eval_list_per_tag])
    pred_strict_incorrect = sum([x['PRED']['strict']['incorrect'] for x in selected_sem_eval_list_per_tag])
    pred_strict_partial = sum([x['PRED']['strict']['partial'] for x in selected_sem_eval_list_per_tag])
    pred_strict_missed = sum([x['PRED']['strict']['missed'] for x in selected_sem_eval_list_per_tag])
    pred_strict_spurious = sum([x['PRED']['strict']['spurious'] for x in selected_sem_eval_list_per_tag])
    pred_strict_possible = sum([x['PRED']['strict']['possible'] for x in selected_sem_eval_list_per_tag])
    pred_strict_actual = sum([x['PRED']['strict']['actual'] for x in selected_sem_eval_list_per_tag])
    pred_strict_precision = statistics.mean([x['PRED']['strict']['precision'] for x in selected_sem_eval_list_per_tag])
    pred_strict_recall = statistics.mean([x['PRED']['strict']['recall'] for x in selected_sem_eval_list_per_tag])
    pred_strict_f1 = statistics.mean([x['PRED']['strict']['f1'] for x in selected_sem_eval_list_per_tag])

    pred_strict_dict = {'Strict': {'Correct': pred_strict_correct, 'Incorrect': pred_strict_incorrect, 'Partial': pred_strict_partial, 'Missed': pred_strict_missed,
                                'Spurious': pred_strict_spurious, 'Possible': pred_strict_possible, 'Actual': pred_strict_actual,
                                'Precision': pred_strict_precision,
                                'Recall': pred_strict_recall, 'F1': pred_strict_f1}}
    all_dict['Scores_per_tag']['Predicates'].update(pred_strict_dict)

    pred_exact_correct = sum([x['PRED']['exact']['correct'] for x in selected_sem_eval_list_per_tag])
    pred_exact_incorrect = sum([x['PRED']['exact']['incorrect'] for x in selected_sem_eval_list_per_tag])
    pred_exact_partial = sum([x['PRED']['exact']['partial'] for x in selected_sem_eval_list_per_tag])
    pred_exact_missed = sum([x['PRED']['exact']['missed'] for x in selected_sem_eval_list_per_tag])
    pred_exact_spurious = sum([x['PRED']['exact']['spurious'] for x in selected_sem_eval_list_per_tag])
    pred_exact_possible = sum([x['PRED']['exact']['possible'] for x in selected_sem_eval_list_per_tag])
    pred_exact_actual = sum([x['PRED']['exact']['actual'] for x in selected_sem_eval_list_per_tag])
    pred_exact_precision = statistics.mean([x['PRED']['exact']['precision'] for x in selected_sem_eval_list_per_tag])
    pred_exact_recall = statistics.mean([x['PRED']['exact']['recall'] for x in selected_sem_eval_list_per_tag])
    pred_exact_f1 = statistics.mean([x['PRED']['exact']['f1'] for x in selected_sem_eval_list_per_tag])

    pred_exact_dict = {'Exact': {'Correct': pred_exact_correct, 'Incorrect': pred_exact_incorrect, 'Partial': pred_exact_partial, 'Missed': pred_exact_missed,
                              'Spurious': pred_exact_spurious, 'Possible': pred_exact_possible, 'Actual': pred_exact_actual,
                              'Precision': pred_exact_precision,
                              'Recall': pred_exact_recall, 'F1': pred_exact_f1}}
    all_dict['Scores_per_tag']['Predicates'].update(pred_exact_dict)

    all_dict['Scores_per_tag'].update({'Objects': {}})

    obj_ent_type_correct = sum([x['OBJ']['ent_type']['correct'] for x in selected_sem_eval_list_per_tag])
    obj_ent_type_incorrect = sum([x['OBJ']['ent_type']['incorrect'] for x in selected_sem_eval_list_per_tag])
    obj_ent_type_partial = sum([x['OBJ']['ent_type']['partial'] for x in selected_sem_eval_list_per_tag])
    obj_ent_type_missed = sum([x['OBJ']['ent_type']['missed'] for x in selected_sem_eval_list_per_tag])
    obj_ent_type_spurious = sum([x['OBJ']['ent_type']['spurious'] for x in selected_sem_eval_list_per_tag])
    obj_ent_type_possible = sum([x['OBJ']['ent_type']['possible'] for x in selected_sem_eval_list_per_tag])
    obj_ent_type_actual = sum([x['OBJ']['ent_type']['actual'] for x in selected_sem_eval_list_per_tag])
    obj_ent_type_precision = statistics.mean([x['OBJ']['ent_type']['precision'] for x in selected_sem_eval_list_per_tag])
    obj_ent_type_recall = statistics.mean([x['OBJ']['ent_type']['recall'] for x in selected_sem_eval_list_per_tag])
    obj_ent_type_f1 = statistics.mean([x['OBJ']['ent_type']['f1'] for x in selected_sem_eval_list_per_tag])

    obj_ent_type_dict = {
        'Ent_type': {'Correct': obj_ent_type_correct, 'Incorrect': obj_ent_type_incorrect, 'Partial': obj_ent_type_partial, 'Missed': obj_ent_type_missed,
                     'Spurious': obj_ent_type_spurious, 'Possible': obj_ent_type_possible, 'Actual': obj_ent_type_actual, 'Precision': obj_ent_type_precision,
                     'Recall': obj_ent_type_recall, 'F1': obj_ent_type_f1}}
    all_dict['Scores_per_tag']['Objects'].update(obj_ent_type_dict)

    obj_partial_correct = sum([x['OBJ']['partial']['correct'] for x in selected_sem_eval_list_per_tag])
    obj_partial_incorrect = sum([x['OBJ']['partial']['incorrect'] for x in selected_sem_eval_list_per_tag])
    obj_partial_partial = sum([x['OBJ']['partial']['partial'] for x in selected_sem_eval_list_per_tag])
    obj_partial_missed = sum([x['OBJ']['partial']['missed'] for x in selected_sem_eval_list_per_tag])
    obj_partial_spurious = sum([x['OBJ']['partial']['spurious'] for x in selected_sem_eval_list_per_tag])
    obj_partial_possible = sum([x['OBJ']['partial']['possible'] for x in selected_sem_eval_list_per_tag])
    obj_partial_actual = sum([x['OBJ']['partial']['actual'] for x in selected_sem_eval_list_per_tag])
    obj_partial_precision = statistics.mean([x['OBJ']['partial']['precision'] for x in selected_sem_eval_list_per_tag])
    obj_partial_recall = statistics.mean([x['OBJ']['partial']['recall'] for x in selected_sem_eval_list_per_tag])
    obj_partial_f1 = statistics.mean([x['OBJ']['partial']['f1'] for x in selected_sem_eval_list_per_tag])

    obj_partial_dict = {
        'Partial': {'Correct': obj_partial_correct, 'Incorrect': obj_partial_incorrect, 'Partial': obj_partial_partial, 'Missed': obj_partial_missed,
                    'Spurious': obj_partial_spurious, 'Possible': obj_partial_possible, 'Actual': obj_partial_actual, 'Precision': obj_partial_precision,
                    'Recall': obj_partial_recall, 'F1': obj_partial_f1}}
    all_dict['Scores_per_tag']['Objects'].update(obj_partial_dict)

    obj_strict_correct = sum([x['OBJ']['strict']['correct'] for x in selected_sem_eval_list_per_tag])
    obj_strict_incorrect = sum([x['OBJ']['strict']['incorrect'] for x in selected_sem_eval_list_per_tag])
    obj_strict_partial = sum([x['OBJ']['strict']['partial'] for x in selected_sem_eval_list_per_tag])
    obj_strict_missed = sum([x['OBJ']['strict']['missed'] for x in selected_sem_eval_list_per_tag])
    obj_strict_spurious = sum([x['OBJ']['strict']['spurious'] for x in selected_sem_eval_list_per_tag])
    obj_strict_possible = sum([x['OBJ']['strict']['possible'] for x in selected_sem_eval_list_per_tag])
    obj_strict_actual = sum([x['OBJ']['strict']['actual'] for x in selected_sem_eval_list_per_tag])
    obj_strict_precision = statistics.mean([x['OBJ']['strict']['precision'] for x in selected_sem_eval_list_per_tag])
    obj_strict_recall = statistics.mean([x['OBJ']['strict']['recall'] for x in selected_sem_eval_list_per_tag])
    obj_strict_f1 = statistics.mean([x['OBJ']['strict']['f1'] for x in selected_sem_eval_list_per_tag])

    obj_strict_dict = {
        'Strict': {'Correct': obj_strict_correct, 'Incorrect': obj_strict_incorrect, 'Partial': obj_strict_partial, 'Missed': obj_strict_missed,
                   'Spurious': obj_strict_spurious, 'Possible': obj_strict_possible, 'Actual': obj_strict_actual,
                   'Precision': obj_strict_precision,
                   'Recall': obj_strict_recall, 'F1': obj_strict_f1}}
    all_dict['Scores_per_tag']['Objects'].update(obj_strict_dict)

    obj_exact_correct = sum([x['OBJ']['exact']['correct'] for x in selected_sem_eval_list_per_tag])
    obj_exact_incorrect = sum([x['OBJ']['exact']['incorrect'] for x in selected_sem_eval_list_per_tag])
    obj_exact_partial = sum([x['OBJ']['exact']['partial'] for x in selected_sem_eval_list_per_tag])
    obj_exact_missed = sum([x['OBJ']['exact']['missed'] for x in selected_sem_eval_list_per_tag])
    obj_exact_spurious = sum([x['OBJ']['exact']['spurious'] for x in selected_sem_eval_list_per_tag])
    obj_exact_possible = sum([x['OBJ']['exact']['possible'] for x in selected_sem_eval_list_per_tag])
    obj_exact_actual = sum([x['OBJ']['exact']['actual'] for x in selected_sem_eval_list_per_tag])
    obj_exact_precision = statistics.mean([x['OBJ']['exact']['precision'] for x in selected_sem_eval_list_per_tag])
    obj_exact_recall = statistics.mean([x['OBJ']['exact']['recall'] for x in selected_sem_eval_list_per_tag])
    obj_exact_f1 = statistics.mean([x['OBJ']['exact']['f1'] for x in selected_sem_eval_list_per_tag])

    obj_exact_dict = {'Exact': {'Correct': obj_exact_correct, 'Incorrect': obj_exact_incorrect, 'Partial': obj_exact_partial, 'Missed': obj_exact_missed,
                               'Spurious': obj_exact_spurious, 'Possible': obj_exact_possible, 'Actual': obj_exact_actual,
                               'Precision': obj_exact_precision,
                               'Recall': obj_exact_recall, 'F1': obj_exact_f1}}
    all_dict['Scores_per_tag']['Objects'].update(obj_exact_dict)

    return all_dict, triple_score, combination_selected, triple_score_sum

def calculateExactTripleScore(ref_list, cand_list, all_dict):
    new_ref_list = [[string.lower() for string in sublist] for sublist in ref_list]
    new_cand_list = [[string.lower() for string in sublist] for sublist in cand_list]
    #First get all the classes by combining the triples in the candidatelist and referencelist
    all_classes = new_cand_list + new_ref_list
    all_classes = [item for items in all_classes for item in items]
    all_classes = list(set(all_classes))

    lb = preprocessing.MultiLabelBinarizer(classes=all_classes)
    mcbin = lb.fit_transform(new_cand_list)
    mrbin = lb.fit_transform(new_ref_list)

    precision = precision_score(mrbin, mcbin, average='macro')
    recall = recall_score(mrbin, mcbin, average='macro')
    f1 = f1_score(mrbin, mcbin, average='macro')

    all_dict.update({'Exact_match': {'Precision': precision, 'Recall': recall, 'F1': f1}})

    return all_dict

def evaluate(input_dataframe, outputfile_overall, outputfile_details):
    allcand_ids, all_text, all_cand_triples, new_cand_list, all_ref_triples, new_ref_list = getCandsAndRefsFromCsv(input_dataframe)
    starting_time = timeit.default_timer()
    print("Start time :",starting_time)
    # For each entry, calculate ALL possible scores for every combination of candidate and reference triple
    total_sem_eval_list, total_sem_eval_list_per_tag = calculateAllScores(new_ref_list, new_cand_list)
    function1_time = timeit.default_timer() 
    print("calculate all score time :", function1_time - starting_time)
    # Get best score for each entry (essentially, try to pick the one that actually matched up the triples correctly)
    all_dict, triple_score, combination_selected, triple_score_sum = calculateSystemScore(total_sem_eval_list, total_sem_eval_list_per_tag, new_ref_list, new_cand_list)
    function2_time = timeit.default_timer() 
    print("calculate all score time :", function2_time - function1_time)
    all_dict2 = calculateExactTripleScore(all_ref_triples, all_cand_triples, all_dict)
    with open(outputfile_overall, 'w') as outfile:
        json.dump(all_dict2, outfile)

    all = {}
    all['id'] = allcand_ids.tolist()
    all['text'] = list(all_text)
    all['ref'] = all_ref_triples
    all['cand'] = all_cand_triples
    all['triple_score'] = triple_score
    all['combination'] = combination_selected
    all['triple_score_sum'] = triple_score_sum
    with open(outputfile_details, 'w') as outfile:
        json.dump(all, outfile)

def main(
    model_path: str = "",
    tok: str = "",
    prompt_template: str = "",
    max_tokens: int = 1024,
    dump: str = "output.pickle",
    pickle: str = "",
    output_path: str = "",
    output_details_path: str = "",
):
    # Main function from benchmark.py
    print(f"Output: {output_path}\nDetails: {output_details_path}")
    if pickle == "":
        df = benchmark(model_path=model_path, tok=tok, max_tokens=max_tokens, dump=dump, prompt_template=prompt_template)
    else:
        output = pd.read_pickle(pickle)
        dt = load_dataset("UofA-LINGO/text_to_triplets")
        df = pd.DataFrame(dt["test"])
        df["gt"] = df["response"]
        df = df.drop(columns=["response"])
        df["model_output"] = [x[0] for x in output.values()]
        #df = df.drop([i for i in range(2,len(df))])
    if output_path == "":
        output_path = 'results/evaluation/llama/vicuna-7b-with-explanasion-test-combined.json'
        print(f"Set default output_path: {output_path}")
    if output_details_path == "":
        output_details_path = 'results/evaluation/llama/vicuna-7b-with-explanasion-test-combined-details.json'
        print(f"Set default output_path: {output_details_path}")
    evaluate(df, output_path, output_details_path)

#main(currentpath + '/Refs.xml', currentpath + '/Cands2.xml', currentpath + '/Results.json')
if __name__ == '__main__':
    fire.Fire(main)
    #main()
    """
    # main(sys.argv[1], sys.argv[2], sys.argv[3])
    # main('Refs.xml', 'Cands2.xml', 'Results.json')
    # ref_file_path = 'data/webnlg_data/release_v3.0/en/test/semantic-parsing-test-data-with-refs-en.xml'
    # cand_file_path = 'results/vocab_dbpedia_triples_with_reverse _20230309-113542_web_nlg_test_50_samples_with_seed_66_num_of_runs_1_rebel.tsv'
    input_file_path = 'results/llama/vicuna-7b-with-explanasion-correct.csv'

    output_path = 'results/evaluation/llama/vicuna-7b-with-explanasion-correct.json'
    output_details_path = 'results/evaluation/llama/vicuna-7b-with-explanasion-correct_details.json'
    

    # main(ref_file_path, cand_file_path,output_path, output_details_path)
    main(input_file_path, output_path, output_details_path)
    """
