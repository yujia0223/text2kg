# Benchmark imports
import pandas as pd  # Import pandas library
import sys
from datasets import load_dataset
import fire
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import pickle
# alpaca-lora utils
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

from tqdm import tqdm

# Evaluation imports
from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning
simplefilter(action='ignore', category=UndefinedMetricWarning)
import os
import regex as re
import statistics
import sys
from nervaluate import Evaluator
import nltk
from nltk.util import ngrams
import string
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import preprocessing
import json
from queue import PriorityQueue, Queue

import numpy as np
import timeit

device = "cuda"
currentpath = os.getcwd()

def benchmark(
    model_path: str = "",  # Path to the model
    tok: str = "",  # Path to tokenizer, will default to model path
    max_tokens: int = 1024,  # Maximum number of tokens the model is allowed to generate
    dump: str = "output.pickle",  # Output file to put raw results into
    load_8bit: bool = False,  # Loads model in 8-bit, use for larger models
    error: str = "errors0.txt",  # File to output any error messages that arise during inference
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    test: str = "UofA-LINGO/text_to_triplets_new_ins"  # Test dataset to load
):
    if model_path == "":
        print("Enter the path to the model. (python benchmark_and_evaluate.py --model_path=/home/tsadler/models/vicuna-7b)")
        exit()
    prompter = Prompter(prompt_template)
    print(f"Benchmarking model at: {model_path}")
    print(f"Using tokenizer at (blank means model_path): {tok}")
    tokenizer = None
    if tok == "":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tok)

    model = AutoModelForCausalLM.from_pretrained(
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
        error="errors0.txt",
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=max_tokens,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        
        # Stop generation at EOS token, add END tag as this caused issues previously.
        eos_tokens = [tokenizer.eos_token_id, tokenizer.encode("### END")[-1]]

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
            try:
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_tokens,
                    pad_token_id=0,
                )
                s = generation_output.sequences[0]
            except ValueError as v:
                with open(error, 'a') as f:
                    print(model_path, '\n', v, '\n', input, '\n', file=f)
                s = torch.tensor([1,2])
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    dt = load_dataset(test)
    output = {}
    for i in tqdm(range(len(dt["test"]))):
        entry = dt["test"][i]
        output[i] = list(evaluate(entry["instruction"], entry["input"], error))
    
    # Write out raw output incase of a crash later on
    with open(dump, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Generate dataframe for the evaluation code
    dt = load_dataset(test)
    df = pd.DataFrame(dt["test"])
    df["gt"] = df["output"]
    df = df.drop(columns=["output"])
    df["model_output"] = [x[0] for x in output.values()]
    return df


def split_ignore_quotes_and_underscore(input_string):
    input_string = input_string.replace(',_', '--PLACEHOLDER--')
    # Pattern splits on commas, ignoring any surrounded by double quotes
    pattern = r',(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^"]*"[^"]*(?:"[^"]*"[^"]*)*$)'
    input_string = re.split(pattern, input_string)
    input_string = [x.replace('--PLACEHOLDER--', ',_') for x in input_string]
    return input_string


def getCandsAndRefs(df):
    print(df.head())

    allcand_ids = df.index.values
    all_text = df['input'].values

    all_cand_triples = []
    all_ref_triples = []
    for i in range(len(df)):
        triples_str_cand = df['model_output'].values[i]
        # Remove EOS token
        triples_str_cand = triples_str_cand.replace('###', '')
        triples_str_cand = triples_str_cand.replace('</s>', '')
        triples_str_cand = triples_str_cand.replace('<|im_end|>', '')

        # New style triples
        triples_cand = triples_str_cand.strip().split('\n')

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
        #triples_cand_tmp = re.findall(r"'(.*?[|].*?[|].*?)'[],]", triples_str_cand)
        #triples_cand = []
        #for triple in triples_cand_tmp:
        #    triple = triple.split(' | ')
        #    triples_cand.append(f'({triple[0]}, {triple[1]}, {triple[2]})')
        tmp = []
        for triple in triples_cand:
            # Remove extra double quotes, triples won't get split properly if everything is in double quotes.
            triple = triple.replace('("', '(')
            if triple.count('"') % 2 == 1:
                triple = triple.replace('")', ')')
            # For splitting on commas, but not those that are surrounded by quotes or those followed by an underscore. Used to properly format.
            t = split_ignore_quotes_and_underscore(triple)
            # Only use comma split if triple is not already delimited by | characters.
            if len(triple.split(' | ')) != 3 and len(t) == 3:
                triple = f'{t[0].strip()} | {t[1].strip()} | {t[2].strip()}' 

            # Do not penalize the model for errors in splitting that cause empty strings
            if triple == '':
                continue
            if triple == '<s>':
                # Can't just use spaces as it won't result in a length three triple due to replacing double spaces
                triple = '( | , | )'
            # To prevent index errors later, pad incomplete triples with empty strings.
            if len(triple.split(' | ')) < 3:
                if len(triple.split(' | ')) == 1:
                    triple += ' | , | )'
                elif len(triple.split(' | ')) == 2:
                    triple += ' | )'
            if len(triple.split(' | ')) > 3:
                print(triple)
                print(triple.split(' | '))
            else:
                tmp.append(triple)
        all_cand_triples.append(tmp)

        triples_str_ref = df['gt'].values[i]
        triples_ref = triples_str_ref.split('\n')
        all_ref_triples.append(triples_ref)

    new_cand_list = []
    
    for entry in all_cand_triples:
        new_triples = []
        for triple in entry:
            # Split camel case words into multiple words.
            new_triple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            # Multiple spaces to single space
            new_triple = re.sub(r'\s+', ' ', new_triple).lower()
            new_triples.append(new_triple)
        new_cand_list.append(new_triples)

    new_ref_list = []
    for entry in all_ref_triples:
        new_triples = []
        for triple in entry:
            # Split camel case words into multiple words.
            new_triple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            # Multiple spaces to single space
            new_triple = re.sub(r'\s+', ' ', new_triple).lower()
            new_triples.append(new_triple)
        new_ref_list.append(new_triples)

    return allcand_ids, all_text, all_cand_triples, new_cand_list, all_ref_triples, new_ref_list


# Finds all elements of ref that match the first element of cand. Then, starting at that index
# if the sublist of ref of length cand all match, returns the start and end indices of ref that represent the
# matching sublist.
def findSubList(cand,ref):
    cand_len=len(cand)
    for ind in (i for i,e in enumerate(ref) if e==cand[0]):
        if ref[ind:ind+cand_len]==cand:
            return ind,ind+cand_len-1

#We are going to try to find matches with the reference, starting with the highest chunk possible (all the words in the reference).
#If we don't find that, we are going to search for all n-grams -1 the number of words in the reference; then -2; then -3; etc.
def nonRefWords(new_ref_list, new_cand_list, foundnum, ngram_length):
    while ngram_length > 0:
        #Get a list of all the ngrams of that size
        ngram_list = list(ngrams(new_cand_list, ngram_length))
        if ngram_list == []:
            print("Empty ngram")
        for ngram in ngram_list:
            #If we find this ngram (in the same order) in the reference
            #We're getting the start and end index of the ngram in the reference
            find_new_ref = findSubList(list(ngram), new_ref_list)
            if find_new_ref is not None:
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

    # Reconstruct any split literal values by ensuring triple lists are no longer than three entries.
    # If they are, check for unmatched double quotes and combine these back into one.
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

    # Only idx is used here
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
            print("candidate:",candidate)
            exit(0)

        # Tokenization generally splits on brackets, for example (turn_me_on_(album) became
        # ['(', 'turn_me_on_', '(', 'album', ')']
        ref_list = nltk.word_tokenize(refsub)
        cand_list = nltk.word_tokenize(candsub)

        # Removing some punctuation / extra characters such as brackets
        # The above becomes ['turn_me_on_', 'album']
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

    evaluator = Evaluator([all_ref_dict], [all_cand_dict], tags=['SUB', 'PRED', 'OBJ'])

    # Returns overall metrics and metrics for each tag

    results, results_per_tag = evaluator.evaluate()

    return results, results_per_tag


def calculateAllScores(new_ref_list, new_cand_list):
    total_sem_eval_list = []
    total_sem_eval_list_per_tag = []

    for idx, candidate in enumerate(new_cand_list):
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


# For verbose output file
def sumAllCombination(selected_sem_eval_list):
    all_dict = {}
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


def collectTripleScores(selected_sem_eval_list, uppercase=False):
    tags = ['correct','incorrect','partial','missed','spurious','possible','actual','precision','recall','f1']
    # From original code, needed to have this function able to return uppercase or lowercase dict keys
    if uppercase:
        tags = ['Correct','Incorrect','Partial','Missed','Spurious','Possible','Actual','Precision','Recall','F1']

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
    ent_type_dict = {'ent_type': {tags[0]: ent_type_correct, tags[1]: ent_type_incorrect, tags[2]: ent_type_partial,
                                tags[3]: ent_type_missed, tags[4]: ent_type_spurious, tags[5]: ent_type_possible,
                                tags[6]: ent_type_actual, tags[7]: ent_type_precision, tags[8]: ent_type_recall,
                                tags[9]: ent_type_f1}}

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
    partial_dict = {'partial': {tags[0]: partial_correct, tags[1]: partial_incorrect, tags[2]: partial_partial,
                                tags[3]: partial_missed, tags[4]: partial_spurious, tags[5]: partial_possible,
                                tags[6]: partial_actual, tags[7]: partial_precision, tags[8]: partial_recall,
                                tags[9]: partial_f1}}

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
    strict_dict = {'strict': {tags[0]: strict_correct, tags[1]: strict_incorrect, tags[2]: strict_partial,
                                tags[3]: strict_missed, tags[4]: strict_spurious, tags[5]: strict_possible,
                                tags[6]: strict_actual, tags[7]: strict_precision, tags[8]: strict_recall,
                                tags[9]: strict_f1}}

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
    exact_dict = {'exact': {tags[0]: exact_correct, tags[1]: exact_incorrect, tags[2]: exact_partial,
                            tags[3]: exact_missed, tags[4]: exact_spurious, tags[5]: exact_possible,
                            tags[6]: exact_actual, tags[7]: exact_precision, tags[8]: exact_recall,
                            tags[9]: exact_f1}}
    if not uppercase:
        collected_dict = {"ent_type": ent_type_dict['ent_type'],
                        "partial": partial_dict['partial'],
                        "exact": exact_dict['exact'],
                        "strict": strict_dict['strict']}
    else:
        collected_dict = {"Ent_type": ent_type_dict['ent_type'],
                        "Partial": partial_dict['partial'],
                        "Exact": exact_dict['exact'],
                        "Strict": strict_dict['strict']}
    return collected_dict


def collectTagScores(selected_sem_eval_list, tag, uppercase=False):
    tags = ['correct','incorrect','partial','missed','spurious','possible','actual','precision','recall','f1']
    # From original code, needed to have this function able to return uppercase or lowercase dict keys
    if uppercase:
        tags = ['Correct','Incorrect','Partial','Missed','Spurious','Possible','Actual','Precision','Recall','F1']

    ent_type_correct = sum([x[tag]['ent_type']['correct'] for x in selected_sem_eval_list])
    ent_type_incorrect = sum([x[tag]['ent_type']['incorrect'] for x in selected_sem_eval_list])
    ent_type_partial = sum([x[tag]['ent_type']['partial'] for x in selected_sem_eval_list])
    ent_type_missed = sum([x[tag]['ent_type']['missed'] for x in selected_sem_eval_list])
    ent_type_spurious = sum([x[tag]['ent_type']['spurious'] for x in selected_sem_eval_list])
    ent_type_possible = sum([x[tag]['ent_type']['possible'] for x in selected_sem_eval_list])
    ent_type_actual = sum([x[tag]['ent_type']['actual'] for x in selected_sem_eval_list])
    ent_type_precision = statistics.mean([x[tag]['ent_type']['precision'] for x in selected_sem_eval_list])
    ent_type_recall = statistics.mean([x[tag]['ent_type']['recall'] for x in selected_sem_eval_list])
    ent_type_f1 = statistics.mean([x[tag]['ent_type']['f1'] for x in selected_sem_eval_list])
    ent_type_dict = {'ent_type': {tags[0]: ent_type_correct, tags[1]: ent_type_incorrect, tags[2]: ent_type_partial,
                                tags[3]: ent_type_missed, tags[4]: ent_type_spurious, tags[5]: ent_type_possible,
                                tags[6]: ent_type_actual, tags[7]: ent_type_precision, tags[8]: ent_type_recall,
                                tags[9]: ent_type_f1}}

    partial_correct = sum([x[tag]['partial']['correct'] for x in selected_sem_eval_list])
    partial_incorrect = sum([x[tag]['partial']['incorrect'] for x in selected_sem_eval_list])
    partial_partial = sum([x[tag]['partial']['partial'] for x in selected_sem_eval_list])
    partial_missed = sum([x[tag]['partial']['missed'] for x in selected_sem_eval_list])
    partial_spurious = sum([x[tag]['partial']['spurious'] for x in selected_sem_eval_list])
    partial_possible = sum([x[tag]['partial']['possible'] for x in selected_sem_eval_list])
    partial_actual = sum([x[tag]['partial']['actual'] for x in selected_sem_eval_list])
    partial_precision = statistics.mean([x[tag]['partial']['precision'] for x in selected_sem_eval_list])
    partial_recall = statistics.mean([x[tag]['partial']['recall'] for x in selected_sem_eval_list])
    partial_f1 = statistics.mean([x[tag]['partial']['f1'] for x in selected_sem_eval_list])
    partial_dict = {'partial': {tags[0]: partial_correct, tags[1]: partial_incorrect, tags[2]: partial_partial,
                                tags[3]: partial_missed, tags[4]: partial_spurious, tags[5]: partial_possible,
                                tags[6]: partial_actual, tags[7]: partial_precision, tags[8]: partial_recall,
                                tags[9]: partial_f1}}

    strict_correct = sum([x[tag]['strict']['correct'] for x in selected_sem_eval_list])
    strict_incorrect = sum([x[tag]['strict']['incorrect'] for x in selected_sem_eval_list])
    strict_partial = sum([x[tag]['strict']['partial'] for x in selected_sem_eval_list])
    strict_missed = sum([x[tag]['strict']['missed'] for x in selected_sem_eval_list])
    strict_spurious = sum([x[tag]['strict']['spurious'] for x in selected_sem_eval_list])
    strict_possible = sum([x[tag]['strict']['possible'] for x in selected_sem_eval_list])
    strict_actual = sum([x[tag]['strict']['actual'] for x in selected_sem_eval_list])
    strict_precision = statistics.mean([x[tag]['strict']['precision'] for x in selected_sem_eval_list])
    strict_recall = statistics.mean([x[tag]['strict']['recall'] for x in selected_sem_eval_list])
    strict_f1 = statistics.mean([x[tag]['strict']['f1'] for x in selected_sem_eval_list])
    strict_dict = {'strict': {tags[0]: strict_correct, tags[1]: strict_incorrect, tags[2]: strict_partial,
                                tags[3]: strict_missed, tags[4]: strict_spurious, tags[5]: strict_possible,
                                tags[6]: strict_actual, tags[7]: strict_precision, tags[8]: strict_recall,
                                tags[9]: strict_f1}}

    exact_correct = sum([x[tag]['exact']['correct'] for x in selected_sem_eval_list])
    exact_incorrect = sum([x[tag]['exact']['incorrect'] for x in selected_sem_eval_list])
    exact_partial = sum([x[tag]['exact']['partial'] for x in selected_sem_eval_list])
    exact_missed = sum([x[tag]['exact']['missed'] for x in selected_sem_eval_list])
    exact_spurious = sum([x[tag]['exact']['spurious'] for x in selected_sem_eval_list])
    exact_possible = sum([x[tag]['exact']['possible'] for x in selected_sem_eval_list])
    exact_actual = sum([x[tag]['exact']['actual'] for x in selected_sem_eval_list])
    exact_precision = statistics.mean([x[tag]['exact']['precision'] for x in selected_sem_eval_list])
    exact_recall = statistics.mean([x[tag]['exact']['recall'] for x in selected_sem_eval_list])
    exact_f1 = statistics.mean([x[tag]['exact']['f1'] for x in selected_sem_eval_list])
    exact_dict = {'exact': {tags[0]: exact_correct, tags[1]: exact_incorrect, tags[2]: exact_partial,
                            tags[3]: exact_missed, tags[4]: exact_spurious, tags[5]: exact_possible,
                            tags[6]: exact_actual, tags[7]: exact_precision, tags[8]: exact_recall,
                            tags[9]: exact_f1}}
    if not uppercase:
        collected_dict = {"ent_type": ent_type_dict['ent_type'],
                        "partial": partial_dict['partial'],
                        "exact": exact_dict['exact'],
                        "strict": strict_dict['strict']}
    else:
        collected_dict = {"Ent_type": ent_type_dict['ent_type'],
                        "Partial": partial_dict['partial'],
                        "Exact": exact_dict['exact'],
                        "Strict": strict_dict['strict']}
    return collected_dict


def calculateSystemScore(total_sem_eval_list, total_sem_eval_list_per_tag, new_ref_list, new_cand_list):
    selected_sem_eval_list = []
    selected_sem_eval_list_per_tag = []
    triple_score = []
    combination_selected = []
    triple_score_sum = []

    prio_cand_list = []
    cands = []
    test = []
    # Add in all items to queue and priority queue
    for i,_ in enumerate(new_cand_list):
        cand_queue = Queue()
        cand_prio = []
        for cand_ind in range(len(new_cand_list[i])):
            prio = PriorityQueue()
            test_tmp = []
            for ref_ind in range(len(new_ref_list[i])):
                collected_scores = total_sem_eval_list[i][cand_ind][ref_ind]
                f1_score = statistics.mean([collected_scores['ent_type']['f1'], collected_scores['partial']['f1'], collected_scores['strict']['f1'], collected_scores['exact']['f1']])
                prio.put((-f1_score, ref_ind), block=False)
                test_tmp.append((-f1_score, ref_ind))
            cand_prio.append(prio)
            cand_queue.put(cand_ind, block=False)
            test.append(test_tmp)
        cands.append(cand_queue)
        prio_cand_list.append(cand_prio)

    # Go through priority lists, extract highest combination. At each step, we take an element from the priority queue,
    # check if it is the highest score seen for the given reference. If it isn't replace with the new score, and add
    # the candidate back. Otherwise, check to see if at the given candidate we could get a higher overall score, by
    # seeing if current element plus top of originally chosen candidate give a higher overall score.
    for i in range(len(cands)):
        total_dict = {'totalscore': 0}
        collected_sem_eval = []
        collected_sem_eval_per_tag = []
        ref_dict = dict()  # Stores ref as key, (F1, cand) as value.
        while(not cands[i].empty()):
            ind = cands[i].get()
            score_ref = prio_cand_list[i][ind].get()
            if score_ref[1] not in ref_dict:
                ref_dict[score_ref[1]] = (-score_ref[0], ind)
            elif ref_dict[score_ref[1]][0] < -score_ref[0]:
                cands[i].put(ref_dict[score_ref[1]][1], block=False)
                ref_dict[score_ref[1]] = (-score_ref[0], ind)
            # Comment out these lines for candidate based greedy approach
            # Subtraction as values in queue are negative
            elif -(prio_cand_list[i][ref_dict[score_ref[1]][1]].queue[0][0]+score_ref[0]) > ref_dict[score_ref[1]][0]-prio_cand_list[i][ind].queue[0][0]:
                cands[i].put(ref_dict[score_ref[1]][1], block=False)
                ref_dict[score_ref[1]] = (-score_ref[0], ind)
            else:
                cands[i].put(ind, block=False)  # Have to keep going until everything is matched
        # Grab out all the combinations we ended up with
        collected_combinations = []
        for j in ref_dict.keys():
            collected_combinations.append([ref_dict[j][1], j])
            collected_sem_eval.append(total_sem_eval_list[i][ref_dict[j][1]][j])
            collected_sem_eval_per_tag.append(total_sem_eval_list_per_tag[i][ref_dict[j][1]][j])
            combi_score = ref_dict[j][0]
            total_dict = {'totalscore': combi_score, 'combination': collected_combinations, 'sem_eval_list': collected_sem_eval,
                        'sem_eval_per_tag_list': collected_sem_eval_per_tag}
        triple_score.append(total_dict['sem_eval_list'])
        combination_selected.append(total_dict['combination'])
        ent_type_dict = sumAllCombination(total_dict['sem_eval_list'])
        triple_score_sum.append(ent_type_dict)
        coll_dict = collectTripleScores(selected_sem_eval_list=total_dict['sem_eval_list'])
        selected_sem_eval_list = selected_sem_eval_list + [coll_dict]
        tag_dict = dict()
        tag_dict['SUB'] = collectTagScores(total_dict['sem_eval_per_tag_list'], 'SUB')
        tag_dict['PRED'] = collectTagScores(total_dict['sem_eval_per_tag_list'], 'PRED')
        tag_dict['OBJ'] = collectTagScores(total_dict['sem_eval_per_tag_list'], 'OBJ')
        selected_sem_eval_list_per_tag = selected_sem_eval_list_per_tag + [tag_dict]

    print("Entries to be averaged / summed:", len([x['ent_type']['precision'] for x in selected_sem_eval_list]))
    all_dict = dict()
    all_dict['Total_scores'] = collectTripleScores(selected_sem_eval_list, True)
    all_dict['Scores_per_tag'] = dict()
    all_dict['Scores_per_tag']['Subjects'] = collectTagScores(selected_sem_eval_list_per_tag, "SUB", True)
    all_dict['Scores_per_tag']['Predicates'] = collectTagScores(selected_sem_eval_list_per_tag, "PRED", True)
    all_dict['Scores_per_tag']['Objects'] = collectTagScores(selected_sem_eval_list_per_tag, "OBJ", True)

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
    allcand_ids, all_text, all_cand_triples, new_cand_list, all_ref_triples, new_ref_list = getCandsAndRefs(input_dataframe)
    starting_time = timeit.default_timer()
    print("Start time :",starting_time)
    # For each entry, calculate ALL possible scores for every combination of candidate and reference triple
    total_sem_eval_list, total_sem_eval_list_per_tag = calculateAllScores(new_ref_list, new_cand_list)
    function1_time = timeit.default_timer() 
    print("calculate all score time :", function1_time - starting_time)
    # Get best score for each entry (essentially, try to pick the one that actually matched up the triples correctly)
    all_dict, triple_score, combination_selected, triple_score_sum = calculateSystemScore(total_sem_eval_list, total_sem_eval_list_per_tag, new_ref_list, new_cand_list)
    function2_time = timeit.default_timer() 
    print("calculate system score time :", function2_time - function1_time)
    all_dict2 = calculateExactTripleScore(all_ref_triples, all_cand_triples, all_dict)
    keys_scores = ['Ent_type', 'Partial', 'Exact', 'Strict']
    keys_metrics = ['Precision', 'Recall', 'F1']
    items = ['Correct', 'Incorrect', 'Partial', 'Missed', 'Spurious', 'Actual', 'Possible']
    for key in keys_scores:
        print(f"For {key}:\nPrecision: {all_dict2['Total_scores'][key][keys_metrics[0]]}     Recall: {all_dict2['Total_scores'][key][keys_metrics[1]]}     F1: {all_dict2['Total_scores'][key][keys_metrics[2]]}\n")
    for key in keys_scores:
        print(f"For {key}:")
        for item in items:
            print(f"{item}: {all_dict2['Total_scores'][key][item]}", end=", ")
        print()
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
    load_8bit: bool = False,
    output_path: str = "",
    output_details_path: str = "",
    error: str = "/home/tsadler/lingo-scripts/errors0.txt",
    test: str = "UofA-LINGO/text_to_triplets_new_ins",
):
    # Main function from benchmark.py
    print(f"Output: {output_path}\nDetails: {output_details_path}")
    if pickle == "":
        df = benchmark(model_path=model_path, tok=tok, max_tokens=max_tokens, dump=dump, prompt_template=prompt_template, error=error, test=test, load_8bit=load_8bit)
    else:
        output = pd.read_pickle(pickle)
        print(len(output.values()))
        dt = load_dataset(test)
        df = pd.DataFrame(dt["test"])
        df["gt"] = df["output"]
        df = df.drop(columns=["output"])
        df["model_output"] = [x[0] for x in output.values()]
    if output_path == "":
        output_path = 'results/evaluation/eval-results.json'
        print(f"Set default output_path: {output_path}")
    if output_details_path == "":
        output_details_path = 'results/evaluation/eval-results-details.json'
        print(f"Set default output_path: {output_details_path}")
    evaluate(df, output_path, output_details_path)


if __name__ == '__main__':
    fire.Fire(main)
