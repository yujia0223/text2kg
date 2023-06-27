# Benchmark imports
import pandas as pd  # Import pandas library
import sys
from datasets import load_dataset
import fire
import torch
import transformers
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# TSadler: For copied dataset
data_files = {"train":"train.csv", "test":"test.csv"}

def benchmark(
    load_8bit: bool = False,
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    csv_file: str = None,  # New argument for CSV file
):
    prompter = Prompter(prompt_template)

    # tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")

    model = LlamaForCausalLM.from_pretrained(
        #"/home/taesiri/src/alpaca-lora/vicuna-7b--based-export-text-to-triplets-explanation-v3/",
        "/home/tsadler/models/lora-vicuna-7b-explanations/hf_ckpt",
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
        max_new_tokens=512,
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

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
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
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    dt = load_dataset("UofA-LINGO/text_to_triplets")
    output = {}
    for i in tqdm(range(len(dt["test"]))):
        entry = dt["test"][i]
        output[i] = list(evaluate(entry["instruction"], entry["context"]))
        # print(output[i])
    # TSadler: Removing intermediate files
    #with open("output-vicuna-7b-with-explanasion-correct.pickle", "wb") as handle:
    #    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TSadler: Removing intermediate CSV file for combined code
    # generate dataframe for the evaluation code
    dt = load_dataset("UofA-LINGO/text_to_triplets")
    df = pd.DataFrame(dt["test"])
    df["gt"] = df["response"]
    df = df.drop(columns=["response"])
    df["model_output"] = [x[0] for x in output.values()]
    return df
    #df.to_csv("vicuna-7b-with-explanasion-correct.csv", index=False)

    # dump df as pickle
    #with open("vicuna-7b-with-explanasion-correct-df.pickle", "wb") as handle:
    #    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)



def get_Cands_and_Refs_from_csv(df):
    #df = pd.read_csv(filepath, header=0)
    print(df.head())

    allcand_ids = df.index.values
    all_text = df['context'].values

    allcandtriples = []
    allreftriples = []
    for i in range(len(df)):
        # newtriples = []
        triples_str_cand = df['model_output'].values[i]

        # vicuna: for this model
        triples_cand = re.findall(r"'(.*?)'", triples_str_cand)
        # print(triples_cand)
        tmp = []
        for triple in triples_cand:
            if len(triple.split(' | ')) != 3:
                continue
            else:
                tmp.append(triple)
        triples_cand = tmp
        # triples_cand = [triple.replace('\\', '').replace(',', '') for triple in triples_cand]
        # triples_cand = ast.literal_eval("[\\" + triples_str_cand + "]")[0]
        # # for triple in triples:
        # #     triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
        # #     newtriples.append(triple_str)
        allcandtriples.append(triples_cand)

        triples_str_ref = df['gt'].values[i]
        triples_ref = ast.literal_eval("[" + triples_str_ref + "]")[0]
        # for triple in triples:
        #     triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
        #     newtriples.append(triple_str)
        allreftriples.append(triples_ref)

    newcandlist = []
    
    for entry in allcandtriples:
        newtriples = []
        # triple 'Turn_Me_On_(album) | runtime | 35.1'
        for triple in entry:
            # triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            adjusttriple = newtriple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = ' | '.join(adjusttriple)
            newtriples.append(newtriple)
        newcandlist.append(newtriples)

    newreflist = []
    
    for entry in allreftriples:
        newtriples = []
        # triple 'Turn_Me_On_(album) | runtime | 35.1'
        for triple in entry:
            # triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            adjusttriple = newtriple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = ' | '.join(adjusttriple)
            newtriples.append(newtriple)
        newreflist.append(newtriples)

    return allcand_ids, all_text, allcandtriples, newcandlist, allreftriples, newreflist

def getRefs(filepath, allcand_ids):
    with open(filepath, encoding='utf-8') as fp:
        refssoup = BeautifulSoup(fp, 'lxml')

    refsentries = refssoup.find('benchmark').find('entries').find_all('entry')

    allreftriples = []
    for index in allcand_ids:
        id = int(index.split('Id')[1])-1
        entry = refsentries[id]
        entryreftriples = []
        modtriplesref = entry.find('modifiedtripleset').find_all('mtriple')
        for modtriple in modtriplesref:
            entryreftriples.append(modtriple.text)
        allreftriples.append(entryreftriples)

    newreflist = []

    for entry in allreftriples:
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            adjusttriple = newtriple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = ' | '.join(adjusttriple)
            newtriples.append(newtriple)
        newreflist.append(newtriples)

    return allreftriples, newreflist

def get_Cands_From_rebel_Tsv(filepath):
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
    allcandtriples = []
    for i in range(len(df)):
        # newtriples = []
        triples_str = df['triples'].values[i]
        triples = ast.literal_eval("[" + triples_str + "]")[0]
        # for triple in triples:
        #     triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
        #     newtriples.append(triple_str)
        allcandtriples.append(triples)

    newcandlist = []
    
    for entry in allcandtriples:
        newtriples = []
        # triple 'Turn_Me_On_(album) | runtime | 35.1'
        for triple in entry:
            # triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            adjusttriple = newtriple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = ' | '.join(adjusttriple)
            newtriples.append(newtriple)
        newcandlist.append(newtriples)

    return allcand_ids, all_text, allcandtriples, newcandlist

def get_Cands_From_Tsv(filepath):
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
    allcandtriples = []
    for i in range(len(df)):
        newtriples = []
        triples_str = df['triples'].values[i]
        triples = ast.literal_eval("[" + triples_str + "]")[0]
        for triple in triples:
            triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
            newtriples.append(triple_str)
        allcandtriples.append(newtriples)

    newcandlist = []
    
    for entry in allcandtriples:
        newtriples = []
        # triple 'Turn_Me_On_(album) | runtime | 35.1'
        for triple in entry:
            # triple_str = triple[0] +' | ' + triple[1] +' | '+ triple[2]
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            adjusttriple = newtriple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = ' | '.join(adjusttriple)
            newtriples.append(newtriple)
        newcandlist.append(newtriples)

    return allcand_ids, all_text, allcandtriples, newcandlist

def getCands(filepath):
    with open(filepath, encoding='utf-8') as fp:
        candssoup = BeautifulSoup(fp, 'lxml')

    candssentries = candssoup.find('benchmark').find('entries').find_all('entry')

    allcandtriples = []

    for entry in candssentries:
        entrycandtriples = []
        modtriplescand = entry.find('generatedtripleset').find_all('gtriple')
        for modtriple in modtriplescand:
            entrycandtriples.append(modtriple.text)
        allcandtriples.append(entrycandtriples)

    newcandlist = []

    for entry in allcandtriples:
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            adjusttriple = newtriple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = ' | '.join(adjusttriple)
            newtriples.append(newtriple)
        newcandlist.append(newtriples)

    return allcandtriples, newcandlist

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

#We are going to try to find matches with the reference, starting with the highest chunk possible (all the words in the reference).
#If we don't find that, we are going to search for all n-grams -1 the number of words in the reference; than -2; than -3; etc.
def nonrefwords(newreflist, newcandlist, foundnum, ngramlength):
    while ngramlength > 0:
        #Get a list of all the ngrams of that size
        ngramlist = list(ngrams(newcandlist, ngramlength))
        for ngram in ngramlist:
            #If we find this ngram (in the same order) in the reference
            if find_sub_list(list(ngram), newreflist) is not None:
                #We're getting the start and end index of the ngram in the reference
                findnewref = find_sub_list(list(ngram), newreflist)
                #And all the numbers in between
                newrefindex = list(range(findnewref[0], findnewref[1] + 1))
                #Change the matched words to FOUNDREF-[FOUNDNUMBER]-[FOUNDINDEX]
                for idx in newrefindex:
                    newreflist[idx] = 'FOUNDREF-' + str(foundnum) + '-' + str(idx)

                #Now find the start and end index of the ngram in the candidate as well
                findnewcand = find_sub_list(list(ngram), newcandlist)
                #And all the indices in between
                newcandindex = list(range(findnewcand[0], findnewcand[1]+1))
                # Change the matched words to FOUNDCAND-[FOUNDNUMBER]-[REFERENCE-FOUNDINDEX]
                for idx, val in enumerate(newcandindex):
                    newcandlist[val] = 'FOUNDCAND-' + str(foundnum) + '-' + str(newrefindex[idx])
                foundnum += 1
                #And try to find new matches again
                nonrefwords(newreflist, newcandlist, foundnum, ngramlength)
        #If no match is found, try to find matches for ngrams 1 smaller
        ngramlength -= 1
    #Return the new lists if all possible ngrams have been searched
    return newreflist, newcandlist

def getrefdict(newreflist, newcandlist, tripletyperef, tripletypecand, baseidx):
    try:
        #If some match is found with the reference
        firstfoundidx = newcandlist.index([i for i in newcandlist if re.findall(r'^FOUNDCAND', i)][0])
        candidatefound = 'y'
    except IndexError:
        candidatefound = 'n'

    if candidatefound == 'y':
        unlinkedlist = []
        beforelist = []
        afterlist = []

        #If the first found candidate match is also the first word in the reference
        if newcandlist[firstfoundidx].endswith('-0'):
            #Flag that some words can appear before the first match, and they are linked with the first candidate match
            beforelinked = 'y'
            firstcand = re.search(r'^(FOUNDCAND-\d+)-', newcandlist[firstfoundidx]).group(1)
        else:
            beforelinked = 'n'

        lastfoundidx = None
        afterlinked = None
        #If there's more words after the last reference, link those to the last reference as well
        #If the last reference word is linked, but the last candidate word is not, one criterion of linking the last words is met
        if (newreflist[-1].startswith('FOUNDREF')) and (not newcandlist[-1].startswith('FOUNDCAND')):
            #If the last linked reference word is the last linked candidate word, the other criterion is also met.
            lastfound = [i for i in newcandlist if re.findall(r'^FOUNDCAND', i)][-1]
            candversion = newreflist[-1].replace('FOUNDREF', 'FOUNDCAND')
            if lastfound == candversion:
                lastfoundidx = newcandlist.index([i for i in newcandlist if re.findall(r'^FOUNDCAND', i)][-1])
                afterlinked = 'y'
                lastcand = re.search(r'^(FOUNDCAND-\d+)-', lastfound).group(1)


        #Ensure that all the not-found blocks are separated by giving them different unlinknumbers
        unlinknumber = 1
        for idx, can in enumerate(newcandlist):
            if not can.startswith('FOUNDCAND'):
                if (idx < firstfoundidx) and (beforelinked == 'y'):
                    newcandlist[idx] = firstcand + '-LINKED'
                    beforelist.append(firstcand + '-LINKED')
                elif (lastfoundidx != None) and (afterlinked != None) and (idx > lastfoundidx) and (afterlinked == 'y'):
                    newcandlist[idx] = lastcand + '-LINKED'
                    afterlist.append(lastcand + '-LINKED')
                else:
                    unlinkedlist.append('NOTFOUND-' + str(unlinknumber))
            else:
                unlinknumber += 1

        totallist = beforelist + newreflist + afterlist + unlinkedlist

        refstart = len(beforelist)
        refend = (len(beforelist) + len(newreflist)) - 1

        refdictlist = [{'label': tripletyperef, 'start': baseidx + refstart, 'end': baseidx + refend}]

        totallist2 = [x.replace('FOUNDREF', 'FOUNDCAND') for x in totallist]

        canddictlist = []
        currentcandidate = ''
        beginidx = ''
        endidx = ''
        collecting = 'n'
        for idx, candidate in enumerate(totallist2):
            if (candidate.startswith('FOUNDCAND')) or (candidate.startswith('NOTFOUND')):
                collecting = 'y'
                curcan = re.search(r'^((.*?)-\d+)', candidate).group(1)
                if curcan != currentcandidate:
                    if currentcandidate != '':
                        endidx = idx-1
                        canddictlist.append({'label': tripletypecand, 'start': baseidx + beginidx, 'end': baseidx + endidx})
                    currentcandidate = curcan
                    beginidx = idx

                if idx == len(totallist2)-1:
                    endidx = idx
                    canddictlist.append({'label': tripletypecand, 'start': baseidx + beginidx, 'end': baseidx + endidx})
            else:
                if collecting == 'y':
                    endidx = idx-1
                    canddictlist.append({'label': tripletypecand, 'start': baseidx + beginidx, 'end': baseidx + endidx})

    else:
        if len(newreflist) == 0:
            refdictlist = []
            canddictlist = [{'label': tripletypecand, 'start': baseidx, 'end': baseidx + (len(newcandlist) - 1)}]
            totallist = newcandlist
        elif len(newcandlist) == 0:
            canddictlist = []
            refdictlist = [{'label': tripletyperef, 'start': baseidx, 'end': baseidx + (len(newreflist) - 1)}]
            totallist = refdictlist
        else:
            totallist = newreflist + newcandlist
            refdictlist = [{'label': tripletyperef, 'start': baseidx, 'end': baseidx + (len(newreflist) - 1)}]
            canddictlist = [{'label': tripletypecand, 'start': baseidx + len(newreflist), 'end': baseidx + (len(totallist) - 1)}]


    return candidatefound, refdictlist, canddictlist, totallist

def evaluaterefcand(reference, candidate):
    newreference = reference.split(' | ')
    newcandidate = candidate.split(' | ')

    #Make sure that reference or candidate aren't '' values originally.
    if (len(newreference) > 1) and (len(newcandidate) > 1):
        indextriple = newreference
    elif (len(newreference) == 1) :
        indextriple = newcandidate
        newreference = ['', '', '']
    else:
        indextriple = newreference
        newcandidate = ['', '', '']

    subjectreflist = None
    subjectcandlist = None
    subjecttotallist = None
    predicatereflist = None
    predicatecandlist = None
    predicatetotallist = None
    objectreflist = None
    objectcandlist = None
    objecttotallist = None
    subjectfound = ''
    predicatefound = ''
    objectfound = ''

    for idx, attrib in enumerate(indextriple):
        #Let's go over each attribute of the triple one by one
        refsub = newreference[idx]
        candsub = newcandidate[idx]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'^[' + re.escape(string.punctuation) + r']+$', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'^[' + re.escape(string.punctuation) + r']+$', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        #Start with an ngram the full number of words in the reference
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        if idx == 0:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'SUB', 'SUB', 0)
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
        elif idx == 1:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'PRED', 'PRED', len(subjecttotallist))
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()
        else:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'OBJ', 'OBJ', len(subjecttotallist) + len(predicatetotallist))
            objectfound = candidatefound
            objectreflist = refdictlist.copy()
            objectcandlist = canddictlist.copy()
            objecttotallist = totallist.copy()

    switchmatchfound = 'n'
    #If no matches were found for two or more attributes, we are going to try and compare different attributes to each other.
    #First let's try to match the candidate subject and reference object (and vice versa)
    if (subjectfound == 'n') and (objectfound == 'n'):
        refsub = newreference[0]
        candsub = newcandidate[2]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'SUB', 'OBJ', 0)

        refsub = newreference[2]
        candsub = newcandidate[0]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(newreflist, newcandlist, 'OBJ', 'SUB', len(totallist) + len(predicatetotallist))

        # subjectfound is based in reference 
        if (candidatefound == 'y') or (candidatefound2 == 'y'):
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
            objectfound = candidatefound2
            objectreflist = refdictlist2.copy()
            objectcandlist = canddictlist2.copy()
            objecttotallist = totallist2.copy()

            # get the new mapping of the predicate
            refpred = newreference[1]
            candpred = newcandidate[1]

            reflist = nltk.word_tokenize(refpred)
            candlist = nltk.word_tokenize(candpred)

            reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
            candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

            newreflist = reflist.copy()
            newcandlist = candlist.copy()
            # Start with an ngram the full number of words in the candidate
            ngramlength = len(newcandlist)
            newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

            # debugging wrong code here nothing to do with predicate
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'PRED', 'PRED', len(subjecttotallist))
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()

            switchmatchfound = 'y'
        else:
            switchmatchfound = 'n'

    # Then, let's try to switch subject and predicate
    if ((subjectfound == 'n') and (predicatefound == 'n')) and (switchmatchfound == 'n'):
        refsub = newreference[0]
        candsub = newcandidate[1]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'SUB', 'PRED', 0)

        refsub = newreference[1]
        candsub = newcandidate[0]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(newreflist, newcandlist, 'PRED', 'SUB', len(totallist))

        if (candidatefound == 'y') or (candidatefound2 == 'y'):
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
            predicatefound = candidatefound2
            predicatereflist = refdictlist2.copy()
            predicatecandlist = canddictlist2.copy()
            predicatetotallist = totallist2.copy()
            switchmatchfound = 'y'
        else:
            switchmatchfound = 'n'

    # Finally, let's try to switch predicate and object
    if ((predicatefound == 'n') and (objectfound == 'n')) and (switchmatchfound == 'n'):
        refsub = newreference[1]
        candsub = newcandidate[2]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'PRED', 'OBJ', len(subjecttotallist))

        refsub = newreference[2]
        candsub = newcandidate[1]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(newreflist, newcandlist, 'OBJ', 'PRED', len(subjecttotallist) + len(totallist))

        if (candidatefound == 'y') or (candidatefound2 == 'y'):
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()
            objectfound = candidatefound2
            objectreflist = refdictlist2.copy()
            objectcandlist = canddictlist2.copy()
            objecttotallist = totallist2.copy()
            switchmatchfound = 'y'
        else:
            switchmatchfound = 'n'


    allrefdict = subjectreflist + predicatereflist + objectreflist
    allcanddict = subjectcandlist + predicatecandlist + objectcandlist
    alltotallist = subjecttotallist + predicatetotallist + objecttotallist

    evaluator = Evaluator([allrefdict], [allcanddict], tags=['SUB', 'PRED', 'OBJ'])

    # Returns overall metrics and metrics for each tag

    results, results_per_tag = evaluator.evaluate()

    return results, results_per_tag

def calculateAllScores(newreflist, newcandlist):
    totalsemevallist = []
    totalsemevallistpertag = []

    for idx, candidate in enumerate(newcandlist):
        print('evaluating candidate ' + str(idx) + ' of ' + str(len(newcandlist)))
        if len(newcandlist[idx]) != len(newreflist[idx]):
            differencebetween = abs(len(newcandlist[idx]) - len(newreflist[idx]))
            differencelist = [''] * differencebetween
            if len(newcandlist[idx]) < len(newreflist[idx]):
                newcandlist[idx] = newcandlist[idx] + differencelist
            else:
                newreflist[idx] = newreflist[idx] + differencelist

    for idx, candidate in enumerate(newcandlist):
        candidatesemeval = []
        candidatesemevalpertag = []
        for triple in candidate:
            triplesemeval = []
            triplesemevalpertag = []
            for reference in newreflist[idx]:
                results, results_per_tag = evaluaterefcand(reference, triple)
                triplesemeval.append(results)
                triplesemevalpertag.append(results_per_tag)

            candidatesemeval.append(triplesemeval)
            candidatesemevalpertag.append(triplesemevalpertag)

        totalsemevallist.append(candidatesemeval)
        totalsemevallistpertag.append(candidatesemevalpertag)

    return totalsemevallist, totalsemevallistpertag

def sum_all_combination(selectedsemevallist):
    # import IPython; IPython.embed()
    alldict = {}
    # alldict.update({'Total_scores': {}})
    for key in selectedsemevallist[0].keys():
        enttypecorrect = sum([x[key]['correct'] for x in selectedsemevallist])
        enttypeincorrect = sum([x[key]['incorrect'] for x in selectedsemevallist])
        enttypepartial = sum([x[key]['partial'] for x in selectedsemevallist])
        enttypemissed = sum([x[key]['missed'] for x in selectedsemevallist])
        enttypespurious = sum([x[key]['spurious'] for x in selectedsemevallist])
        enttypepossible = sum([x[key]['possible'] for x in selectedsemevallist])
        enttypeactual = sum([x[key]['actual'] for x in selectedsemevallist])
        enttypeprecision = statistics.mean([x[key]['precision'] for x in selectedsemevallist])
        enttyperecall = statistics.mean([x[key]['recall'] for x in selectedsemevallist])
        enttypef1 = statistics.mean([x[key]['f1'] for x in selectedsemevallist])

        enttypedict = {key: {'Correct': enttypecorrect, 'Incorrect': enttypeincorrect, 'Partial': enttypepartial, 'Missed': enttypemissed,
                                    'Spurious': enttypespurious, 'Possible': enttypepossible, 'Actual': enttypeactual, 'Precision': enttypeprecision,
                                    'Recall': enttyperecall, 'F1': enttypef1}}
        alldict.update(enttypedict)
    return alldict

def calculateSystemScore(totalsemevallist, totalsemevallistpertag, newreflist, newcandlist):
    selectedsemevallist = []
    selectedsemevallistpertag = []
    triple_score = []
    combination_selected = []
    triple_score_sum = []
    alldicts = []

    # Get all the permutations of the number of scores given per candidate, so if there's 4 candidates, but 3 references, this part ensures that one of
    # The four will not be scored
    for idx, candidate in enumerate(newcandlist):
        print('calculating system score for candidate ' + str(idx) + ' of ' + str(len(newcandlist)))
        # if len(newcandlist[idx]) > len(newreflist[idx]):
        #     # Get all permutations
        #     choosecands = list(itertools.permutations([x[0] for x in enumerate(totalsemevallist[idx])], len(totalsemevallist[idx][0])))
        #     # The permutations in different orders are not necessary: we only need one order without the number of candidates we're looking at
        #     choosecands = set([tuple(sorted(i)) for i in choosecands])  # Sort inner list and then use set
        #     choosecands = list(map(list, choosecands))  # Converting back to list
        # else:
        #     # Otherwise, we're just going to score all candidates
        #     choosecands = [list(range(len(newcandlist[idx])))]

        # # Get all permutations in which the scores can be combined
        # if len(newcandlist[idx]) > len(newreflist[idx]):
        #     choosescore = list(itertools.permutations([x[0] for x in enumerate(totalsemevallist[idx][0])], len(newreflist[idx])))
        #     choosescore = [list(x) for x in choosescore]
        # else:
        #     choosescore = list(itertools.permutations([x[0] for x in enumerate(totalsemevallist[idx][0])], len(newcandlist[idx])))
        #     choosescore = [list(x) for x in choosescore]

        # # Get all possible combinations between the candidates and the scores
        # combilist = list(itertools.product(choosecands, choosescore))

        totaldict = {'totalscore': 0}

        # for combination in combilist:
        #     combiscore = 0
        #     # Take the combination between the candidate and the score
        #     zipcombi = list(zip(combination[0], combination[1]))
        #     collectedsemeval = []
        #     collectedsemevalpertag = []

        #     for zc in zipcombi:
        #         collectedscores = totalsemevallist[idx][zc[0]][zc[1]]
        #         f1score = statistics.mean([collectedscores['ent_type']['f1'], collectedscores['partial']['f1'], collectedscores['strict']['f1'], collectedscores['exact']['f1']])
        #         combiscore += f1score

        #         collectedsemeval.append(collectedscores)
        #         collectedsemevalpertag.append(totalsemevallistpertag[idx][zc[0]][zc[1]])


        #     # If the combination is the highest score thus far, or the first score, make it the totaldict
        #     if (combiscore > totaldict['totalscore']) or (len(totaldict) == 1):
        #         totaldict = {'totalscore': combiscore, 'combination': combination, 'semevallist': collectedsemeval,
        #                      'semevalpertaglist': collectedsemevalpertag}
        # triple_score.append(totaldict['semevallist'])
        # combination_selected.append(totaldict['combination'])
        # enttypedict = sum_all_combination(totaldict['semevallist'])
        # triple_score_sum.append(enttypedict)
        # selectedsemevallist = selectedsemevallist + totaldict['semevallist']
        # selectedsemevallistpertag = selectedsemevallistpertag + totaldict['semevalpertaglist']
        collectedsemeval = []
        collectedsemevalpertag = []
        collected_combinations = []
        for ind_cand in range(len(newcandlist[idx])):
            combiscore = 0
            # Take the combination between the candidate and the score
            f1scores = []
            combination = []
            
            for ind_ref in range(len(newreflist[idx])):
                collectedscores = totalsemevallist[idx][ind_cand][ind_ref]
                f1score = statistics.mean([collectedscores['ent_type']['f1'], collectedscores['partial']['f1'], collectedscores['strict']['f1'], collectedscores['exact']['f1']])
                f1scores.append(f1score)
                combination.append([ind_cand, ind_ref])

            # If the combination is the highest score thus far, or the first score, make it the totaldict
            index_max = np.argmax(f1scores)
            selected_combination = combination[index_max]
            collected_combinations.append(selected_combination)
            collectedsemeval.append(totalsemevallist[idx][selected_combination[0]][selected_combination[1]])
            collectedsemevalpertag.append(totalsemevallistpertag[idx][selected_combination[0]][selected_combination[1]])
            combiscore = f1score
            totaldict = {'totalscore': combiscore, 'combination': collected_combinations, 'semevallist': collectedsemeval,
                        'semevalpertaglist': collectedsemevalpertag}

        triple_score.append(totaldict['semevallist'])
        combination_selected.append(totaldict['combination'])
        enttypedict = sum_all_combination(totaldict['semevallist'])
        triple_score_sum.append(enttypedict)
        selectedsemevallist = selectedsemevallist + totaldict['semevallist']
        selectedsemevallistpertag = selectedsemevallistpertag + totaldict['semevalpertaglist']

    alldict = {}
    alldict.update({'Total_scores': {}})

    enttypecorrect = sum([x['ent_type']['correct'] for x in selectedsemevallist])
    enttypeincorrect = sum([x['ent_type']['incorrect'] for x in selectedsemevallist])
    enttypepartial = sum([x['ent_type']['partial'] for x in selectedsemevallist])
    enttypemissed = sum([x['ent_type']['missed'] for x in selectedsemevallist])
    enttypespurious = sum([x['ent_type']['spurious'] for x in selectedsemevallist])
    enttypepossible = sum([x['ent_type']['possible'] for x in selectedsemevallist])
    enttypeactual = sum([x['ent_type']['actual'] for x in selectedsemevallist])
    enttypeprecision = statistics.mean([x['ent_type']['precision'] for x in selectedsemevallist])
    enttyperecall = statistics.mean([x['ent_type']['recall'] for x in selectedsemevallist])
    enttypef1 = statistics.mean([x['ent_type']['f1'] for x in selectedsemevallist])

    enttypedict = {'Ent_type': {'Correct': enttypecorrect, 'Incorrect': enttypeincorrect, 'Partial': enttypepartial, 'Missed': enttypemissed,
                                'Spurious': enttypespurious, 'Possible': enttypepossible, 'Actual': enttypeactual, 'Precision': enttypeprecision,
                                'Recall': enttyperecall, 'F1': enttypef1}}

    alldict['Total_scores'].update(enttypedict)

    partialcorrect = sum([x['partial']['correct'] for x in selectedsemevallist])
    partialincorrect = sum([x['partial']['incorrect'] for x in selectedsemevallist])
    partialpartial = sum([x['partial']['partial'] for x in selectedsemevallist])
    partialmissed = sum([x['partial']['missed'] for x in selectedsemevallist])
    partialspurious = sum([x['partial']['spurious'] for x in selectedsemevallist])
    partialpossible = sum([x['partial']['possible'] for x in selectedsemevallist])
    partialactual = sum([x['partial']['actual'] for x in selectedsemevallist])
    partialprecision = statistics.mean([x['partial']['precision'] for x in selectedsemevallist])
    partialrecall = statistics.mean([x['partial']['recall'] for x in selectedsemevallist])
    partialf1 = statistics.mean([x['partial']['f1'] for x in selectedsemevallist])

    partialdict = {'Partial': {'Correct': partialcorrect, 'Incorrect': partialincorrect, 'Partial': partialpartial, 'Missed': partialmissed,
                                'Spurious': partialspurious, 'Possible': partialpossible, 'Actual': partialactual, 'Precision': partialprecision,
                                'Recall': partialrecall, 'F1': partialf1}}
    alldict['Total_scores'].update(partialdict)

    strictcorrect = sum([x['strict']['correct'] for x in selectedsemevallist])
    strictincorrect = sum([x['strict']['incorrect'] for x in selectedsemevallist])
    strictpartial = sum([x['strict']['partial'] for x in selectedsemevallist])
    strictmissed = sum([x['strict']['missed'] for x in selectedsemevallist])
    strictspurious = sum([x['strict']['spurious'] for x in selectedsemevallist])
    strictpossible = sum([x['strict']['possible'] for x in selectedsemevallist])
    strictactual = sum([x['strict']['actual'] for x in selectedsemevallist])
    strictprecision = statistics.mean([x['strict']['precision'] for x in selectedsemevallist])
    strictrecall = statistics.mean([x['strict']['recall'] for x in selectedsemevallist])
    strictf1 = statistics.mean([x['strict']['f1'] for x in selectedsemevallist])

    strictdict = {'Strict': {'Correct': strictcorrect, 'Incorrect': strictincorrect, 'Partial': strictpartial, 'Missed': strictmissed,
                                'Spurious': strictspurious, 'Possible': strictpossible, 'Actual': strictactual, 'Precision': strictprecision,
                                'Recall': strictrecall, 'F1': strictf1}}
    alldict['Total_scores'].update(strictdict)

    exactcorrect = sum([x['exact']['correct'] for x in selectedsemevallist])
    exactincorrect = sum([x['exact']['incorrect'] for x in selectedsemevallist])
    exactpartial = sum([x['exact']['partial'] for x in selectedsemevallist])
    exactmissed = sum([x['exact']['missed'] for x in selectedsemevallist])
    exactspurious = sum([x['exact']['spurious'] for x in selectedsemevallist])
    exactpossible = sum([x['exact']['possible'] for x in selectedsemevallist])
    exactactual = sum([x['exact']['actual'] for x in selectedsemevallist])
    exactprecision = statistics.mean([x['exact']['precision'] for x in selectedsemevallist])
    exactrecall = statistics.mean([x['exact']['recall'] for x in selectedsemevallist])
    exactf1 = statistics.mean([x['exact']['f1'] for x in selectedsemevallist])

    exactdict = {'Exact': {'Correct': exactcorrect, 'Incorrect': exactincorrect, 'Partial': exactpartial, 'Missed': exactmissed,
                                'Spurious': exactspurious, 'Possible': exactpossible, 'Actual': exactactual, 'Precision': exactprecision,
                                'Recall': exactrecall, 'F1': exactf1}}
    alldict['Total_scores'].update(exactdict)

    alldict.update({'Scores_per_tag': {}})

    alldict['Scores_per_tag'].update({'Subjects': {}})

    subenttypecorrect = sum([x['SUB']['ent_type']['correct'] for x in selectedsemevallistpertag])
    subenttypeincorrect = sum([x['SUB']['ent_type']['incorrect'] for x in selectedsemevallistpertag])
    subenttypepartial = sum([x['SUB']['ent_type']['partial'] for x in selectedsemevallistpertag])
    subenttypemissed = sum([x['SUB']['ent_type']['missed'] for x in selectedsemevallistpertag])
    subenttypespurious = sum([x['SUB']['ent_type']['spurious'] for x in selectedsemevallistpertag])
    subenttypepossible = sum([x['SUB']['ent_type']['possible'] for x in selectedsemevallistpertag])
    subenttypeactual = sum([x['SUB']['ent_type']['actual'] for x in selectedsemevallistpertag])
    subenttypeprecision = statistics.mean([x['SUB']['ent_type']['precision'] for x in selectedsemevallistpertag])
    subenttyperecall = statistics.mean([x['SUB']['ent_type']['recall'] for x in selectedsemevallistpertag])
    subenttypef1 = statistics.mean([x['SUB']['ent_type']['f1'] for x in selectedsemevallistpertag])

    subenttypedict = {'Ent_type': {'Correct': subenttypecorrect, 'Incorrect': subenttypeincorrect, 'Partial': subenttypepartial, 'Missed': subenttypemissed,
                           'Spurious': subenttypespurious, 'Possible': subenttypepossible, 'Actual': subenttypeactual, 'Precision': subenttypeprecision,
                           'Recall': subenttyperecall, 'F1': subenttypef1}}
    alldict['Scores_per_tag']['Subjects'].update(subenttypedict)

    subpartialcorrect = sum([x['SUB']['partial']['correct'] for x in selectedsemevallistpertag])
    subpartialincorrect = sum([x['SUB']['partial']['incorrect'] for x in selectedsemevallistpertag])
    subpartialpartial = sum([x['SUB']['partial']['partial'] for x in selectedsemevallistpertag])
    subpartialmissed = sum([x['SUB']['partial']['missed'] for x in selectedsemevallistpertag])
    subpartialspurious = sum([x['SUB']['partial']['spurious'] for x in selectedsemevallistpertag])
    subpartialpossible = sum([x['SUB']['partial']['possible'] for x in selectedsemevallistpertag])
    subpartialactual = sum([x['SUB']['partial']['actual'] for x in selectedsemevallistpertag])
    subpartialprecision = statistics.mean([x['SUB']['partial']['precision'] for x in selectedsemevallistpertag])
    subpartialrecall = statistics.mean([x['SUB']['partial']['recall'] for x in selectedsemevallistpertag])
    subpartialf1 = statistics.mean([x['SUB']['partial']['f1'] for x in selectedsemevallistpertag])

    subpartialdict = {'Partial': {'Correct': subpartialcorrect, 'Incorrect': subpartialincorrect, 'Partial': subpartialpartial, 'Missed': subpartialmissed,
                           'Spurious': subpartialspurious, 'Possible': subpartialpossible, 'Actual': subpartialactual, 'Precision': subpartialprecision,
                           'Recall': subpartialrecall, 'F1': subpartialf1}}
    alldict['Scores_per_tag']['Subjects'].update(subpartialdict)

    substrictcorrect = sum([x['SUB']['strict']['correct'] for x in selectedsemevallistpertag])
    substrictincorrect = sum([x['SUB']['strict']['incorrect'] for x in selectedsemevallistpertag])
    substrictpartial = sum([x['SUB']['strict']['partial'] for x in selectedsemevallistpertag])
    substrictmissed = sum([x['SUB']['strict']['missed'] for x in selectedsemevallistpertag])
    substrictspurious = sum([x['SUB']['strict']['spurious'] for x in selectedsemevallistpertag])
    substrictpossible = sum([x['SUB']['strict']['possible'] for x in selectedsemevallistpertag])
    substrictactual = sum([x['SUB']['strict']['actual'] for x in selectedsemevallistpertag])
    substrictprecision = statistics.mean([x['SUB']['strict']['precision'] for x in selectedsemevallistpertag])
    substrictrecall = statistics.mean([x['SUB']['strict']['recall'] for x in selectedsemevallistpertag])
    substrictf1 = statistics.mean([x['SUB']['strict']['f1'] for x in selectedsemevallistpertag])

    substrictdict = {'Strict': {'Correct': substrictcorrect, 'Incorrect': substrictincorrect, 'Partial': substrictpartial, 'Missed': substrictmissed,
                           'Spurious': substrictspurious, 'Possible': substrictpossible, 'Actual': substrictactual, 'Precision': substrictprecision,
                           'Recall': substrictrecall, 'F1': substrictf1}}
    alldict['Scores_per_tag']['Subjects'].update(substrictdict)

    subexactcorrect = sum([x['SUB']['exact']['correct'] for x in selectedsemevallistpertag])
    subexactincorrect = sum([x['SUB']['exact']['incorrect'] for x in selectedsemevallistpertag])
    subexactpartial = sum([x['SUB']['exact']['partial'] for x in selectedsemevallistpertag])
    subexactmissed = sum([x['SUB']['exact']['missed'] for x in selectedsemevallistpertag])
    subexactspurious = sum([x['SUB']['exact']['spurious'] for x in selectedsemevallistpertag])
    subexactpossible = sum([x['SUB']['exact']['possible'] for x in selectedsemevallistpertag])
    subexactactual = sum([x['SUB']['exact']['actual'] for x in selectedsemevallistpertag])
    subexactprecision = statistics.mean([x['SUB']['exact']['precision'] for x in selectedsemevallistpertag])
    subexactrecall = statistics.mean([x['SUB']['exact']['recall'] for x in selectedsemevallistpertag])
    subexactf1 = statistics.mean([x['SUB']['exact']['f1'] for x in selectedsemevallistpertag])

    subexactdict = {'Exact': {'Correct': subexactcorrect, 'Incorrect': subexactincorrect, 'Partial': subexactpartial, 'Missed': subexactmissed,
                                'Spurious': subexactspurious, 'Possible': subexactpossible, 'Actual': subexactactual,
                                'Precision': subexactprecision,
                                'Recall': subexactrecall, 'F1': subexactf1}}
    alldict['Scores_per_tag']['Subjects'].update(subexactdict)

    alldict['Scores_per_tag'].update({'Predicates': {}})

    predenttypecorrect = sum([x['PRED']['ent_type']['correct'] for x in selectedsemevallistpertag])
    predenttypeincorrect = sum([x['PRED']['ent_type']['incorrect'] for x in selectedsemevallistpertag])
    predenttypepartial = sum([x['PRED']['ent_type']['partial'] for x in selectedsemevallistpertag])
    predenttypemissed = sum([x['PRED']['ent_type']['missed'] for x in selectedsemevallistpertag])
    predenttypespurious = sum([x['PRED']['ent_type']['spurious'] for x in selectedsemevallistpertag])
    predenttypepossible = sum([x['PRED']['ent_type']['possible'] for x in selectedsemevallistpertag])
    predenttypeactual = sum([x['PRED']['ent_type']['actual'] for x in selectedsemevallistpertag])
    predenttypeprecision = statistics.mean([x['PRED']['ent_type']['precision'] for x in selectedsemevallistpertag])
    predenttyperecall = statistics.mean([x['PRED']['ent_type']['recall'] for x in selectedsemevallistpertag])
    predenttypef1 = statistics.mean([x['PRED']['ent_type']['f1'] for x in selectedsemevallistpertag])

    predenttypedict = {
        'Ent_type': {'Correct': predenttypecorrect, 'Incorrect': predenttypeincorrect, 'Partial': predenttypepartial, 'Missed': predenttypemissed,
                     'Spurious': predenttypespurious, 'Possible': predenttypepossible, 'Actual': predenttypeactual, 'Precision': predenttypeprecision,
                     'Recall': predenttyperecall, 'F1': predenttypef1}}
    alldict['Scores_per_tag']['Predicates'].update(predenttypedict)

    predpartialcorrect = sum([x['PRED']['partial']['correct'] for x in selectedsemevallistpertag])
    predpartialincorrect = sum([x['PRED']['partial']['incorrect'] for x in selectedsemevallistpertag])
    predpartialpartial = sum([x['PRED']['partial']['partial'] for x in selectedsemevallistpertag])
    predpartialmissed = sum([x['PRED']['partial']['missed'] for x in selectedsemevallistpertag])
    predpartialspurious = sum([x['PRED']['partial']['spurious'] for x in selectedsemevallistpertag])
    predpartialpossible = sum([x['PRED']['partial']['possible'] for x in selectedsemevallistpertag])
    predpartialactual = sum([x['PRED']['partial']['actual'] for x in selectedsemevallistpertag])
    predpartialprecision = statistics.mean([x['PRED']['partial']['precision'] for x in selectedsemevallistpertag])
    predpartialrecall = statistics.mean([x['PRED']['partial']['recall'] for x in selectedsemevallistpertag])
    predpartialf1 = statistics.mean([x['PRED']['partial']['f1'] for x in selectedsemevallistpertag])

    predpartialdict = {
        'Partial': {'Correct': predpartialcorrect, 'Incorrect': predpartialincorrect, 'Partial': predpartialpartial, 'Missed': predpartialmissed,
                    'Spurious': predpartialspurious, 'Possible': predpartialpossible, 'Actual': predpartialactual, 'Precision': predpartialprecision,
                    'Recall': predpartialrecall, 'F1': predpartialf1}}
    alldict['Scores_per_tag']['Predicates'].update(predpartialdict)

    predstrictcorrect = sum([x['PRED']['strict']['correct'] for x in selectedsemevallistpertag])
    predstrictincorrect = sum([x['PRED']['strict']['incorrect'] for x in selectedsemevallistpertag])
    predstrictpartial = sum([x['PRED']['strict']['partial'] for x in selectedsemevallistpertag])
    predstrictmissed = sum([x['PRED']['strict']['missed'] for x in selectedsemevallistpertag])
    predstrictspurious = sum([x['PRED']['strict']['spurious'] for x in selectedsemevallistpertag])
    predstrictpossible = sum([x['PRED']['strict']['possible'] for x in selectedsemevallistpertag])
    predstrictactual = sum([x['PRED']['strict']['actual'] for x in selectedsemevallistpertag])
    predstrictprecision = statistics.mean([x['PRED']['strict']['precision'] for x in selectedsemevallistpertag])
    predstrictrecall = statistics.mean([x['PRED']['strict']['recall'] for x in selectedsemevallistpertag])
    predstrictf1 = statistics.mean([x['PRED']['strict']['f1'] for x in selectedsemevallistpertag])

    predstrictdict = {'Strict': {'Correct': predstrictcorrect, 'Incorrect': predstrictincorrect, 'Partial': predstrictpartial, 'Missed': predstrictmissed,
                                'Spurious': predstrictspurious, 'Possible': predstrictpossible, 'Actual': predstrictactual,
                                'Precision': predstrictprecision,
                                'Recall': predstrictrecall, 'F1': predstrictf1}}
    alldict['Scores_per_tag']['Predicates'].update(predstrictdict)

    predexactcorrect = sum([x['PRED']['exact']['correct'] for x in selectedsemevallistpertag])
    predexactincorrect = sum([x['PRED']['exact']['incorrect'] for x in selectedsemevallistpertag])
    predexactpartial = sum([x['PRED']['exact']['partial'] for x in selectedsemevallistpertag])
    predexactmissed = sum([x['PRED']['exact']['missed'] for x in selectedsemevallistpertag])
    predexactspurious = sum([x['PRED']['exact']['spurious'] for x in selectedsemevallistpertag])
    predexactpossible = sum([x['PRED']['exact']['possible'] for x in selectedsemevallistpertag])
    predexactactual = sum([x['PRED']['exact']['actual'] for x in selectedsemevallistpertag])
    predexactprecision = statistics.mean([x['PRED']['exact']['precision'] for x in selectedsemevallistpertag])
    predexactrecall = statistics.mean([x['PRED']['exact']['recall'] for x in selectedsemevallistpertag])
    predexactf1 = statistics.mean([x['PRED']['exact']['f1'] for x in selectedsemevallistpertag])

    predexactdict = {'Exact': {'Correct': predexactcorrect, 'Incorrect': predexactincorrect, 'Partial': predexactpartial, 'Missed': predexactmissed,
                              'Spurious': predexactspurious, 'Possible': predexactpossible, 'Actual': predexactactual,
                              'Precision': predexactprecision,
                              'Recall': predexactrecall, 'F1': predexactf1}}
    alldict['Scores_per_tag']['Predicates'].update(predexactdict)

    alldict['Scores_per_tag'].update({'Objects': {}})

    objenttypecorrect = sum([x['OBJ']['ent_type']['correct'] for x in selectedsemevallistpertag])
    objenttypeincorrect = sum([x['OBJ']['ent_type']['incorrect'] for x in selectedsemevallistpertag])
    objenttypepartial = sum([x['OBJ']['ent_type']['partial'] for x in selectedsemevallistpertag])
    objenttypemissed = sum([x['OBJ']['ent_type']['missed'] for x in selectedsemevallistpertag])
    objenttypespurious = sum([x['OBJ']['ent_type']['spurious'] for x in selectedsemevallistpertag])
    objenttypepossible = sum([x['OBJ']['ent_type']['possible'] for x in selectedsemevallistpertag])
    objenttypeactual = sum([x['OBJ']['ent_type']['actual'] for x in selectedsemevallistpertag])
    objenttypeprecision = statistics.mean([x['OBJ']['ent_type']['precision'] for x in selectedsemevallistpertag])
    objenttyperecall = statistics.mean([x['OBJ']['ent_type']['recall'] for x in selectedsemevallistpertag])
    objenttypef1 = statistics.mean([x['OBJ']['ent_type']['f1'] for x in selectedsemevallistpertag])

    objenttypedict = {
        'Ent_type': {'Correct': objenttypecorrect, 'Incorrect': objenttypeincorrect, 'Partial': objenttypepartial, 'Missed': objenttypemissed,
                     'Spurious': objenttypespurious, 'Possible': objenttypepossible, 'Actual': objenttypeactual, 'Precision': objenttypeprecision,
                     'Recall': objenttyperecall, 'F1': objenttypef1}}
    alldict['Scores_per_tag']['Objects'].update(objenttypedict)

    objpartialcorrect = sum([x['OBJ']['partial']['correct'] for x in selectedsemevallistpertag])
    objpartialincorrect = sum([x['OBJ']['partial']['incorrect'] for x in selectedsemevallistpertag])
    objpartialpartial = sum([x['OBJ']['partial']['partial'] for x in selectedsemevallistpertag])
    objpartialmissed = sum([x['OBJ']['partial']['missed'] for x in selectedsemevallistpertag])
    objpartialspurious = sum([x['OBJ']['partial']['spurious'] for x in selectedsemevallistpertag])
    objpartialpossible = sum([x['OBJ']['partial']['possible'] for x in selectedsemevallistpertag])
    objpartialactual = sum([x['OBJ']['partial']['actual'] for x in selectedsemevallistpertag])
    objpartialprecision = statistics.mean([x['OBJ']['partial']['precision'] for x in selectedsemevallistpertag])
    objpartialrecall = statistics.mean([x['OBJ']['partial']['recall'] for x in selectedsemevallistpertag])
    objpartialf1 = statistics.mean([x['OBJ']['partial']['f1'] for x in selectedsemevallistpertag])

    objpartialdict = {
        'Partial': {'Correct': objpartialcorrect, 'Incorrect': objpartialincorrect, 'Partial': objpartialpartial, 'Missed': objpartialmissed,
                    'Spurious': objpartialspurious, 'Possible': objpartialpossible, 'Actual': objpartialactual, 'Precision': objpartialprecision,
                    'Recall': objpartialrecall, 'F1': objpartialf1}}
    alldict['Scores_per_tag']['Objects'].update(objpartialdict)

    objstrictcorrect = sum([x['OBJ']['strict']['correct'] for x in selectedsemevallistpertag])
    objstrictincorrect = sum([x['OBJ']['strict']['incorrect'] for x in selectedsemevallistpertag])
    objstrictpartial = sum([x['OBJ']['strict']['partial'] for x in selectedsemevallistpertag])
    objstrictmissed = sum([x['OBJ']['strict']['missed'] for x in selectedsemevallistpertag])
    objstrictspurious = sum([x['OBJ']['strict']['spurious'] for x in selectedsemevallistpertag])
    objstrictpossible = sum([x['OBJ']['strict']['possible'] for x in selectedsemevallistpertag])
    objstrictactual = sum([x['OBJ']['strict']['actual'] for x in selectedsemevallistpertag])
    objstrictprecision = statistics.mean([x['OBJ']['strict']['precision'] for x in selectedsemevallistpertag])
    objstrictrecall = statistics.mean([x['OBJ']['strict']['recall'] for x in selectedsemevallistpertag])
    objstrictf1 = statistics.mean([x['OBJ']['strict']['f1'] for x in selectedsemevallistpertag])

    objstrictdict = {
        'Strict': {'Correct': objstrictcorrect, 'Incorrect': objstrictincorrect, 'Partial': objstrictpartial, 'Missed': objstrictmissed,
                   'Spurious': objstrictspurious, 'Possible': objstrictpossible, 'Actual': objstrictactual,
                   'Precision': objstrictprecision,
                   'Recall': objstrictrecall, 'F1': objstrictf1}}
    alldict['Scores_per_tag']['Objects'].update(objstrictdict)

    objexactcorrect = sum([x['OBJ']['exact']['correct'] for x in selectedsemevallistpertag])
    objexactincorrect = sum([x['OBJ']['exact']['incorrect'] for x in selectedsemevallistpertag])
    objexactpartial = sum([x['OBJ']['exact']['partial'] for x in selectedsemevallistpertag])
    objexactmissed = sum([x['OBJ']['exact']['missed'] for x in selectedsemevallistpertag])
    objexactspurious = sum([x['OBJ']['exact']['spurious'] for x in selectedsemevallistpertag])
    objexactpossible = sum([x['OBJ']['exact']['possible'] for x in selectedsemevallistpertag])
    objexactactual = sum([x['OBJ']['exact']['actual'] for x in selectedsemevallistpertag])
    objexactprecision = statistics.mean([x['OBJ']['exact']['precision'] for x in selectedsemevallistpertag])
    objexactrecall = statistics.mean([x['OBJ']['exact']['recall'] for x in selectedsemevallistpertag])
    objexactf1 = statistics.mean([x['OBJ']['exact']['f1'] for x in selectedsemevallistpertag])

    objexactdict = {'Exact': {'Correct': objexactcorrect, 'Incorrect': objexactincorrect, 'Partial': objexactpartial, 'Missed': objexactmissed,
                               'Spurious': objexactspurious, 'Possible': objexactpossible, 'Actual': objexactactual,
                               'Precision': objexactprecision,
                               'Recall': objexactrecall, 'F1': objexactf1}}
    alldict['Scores_per_tag']['Objects'].update(objexactdict)

    return alldict, triple_score, combination_selected, triple_score_sum

def calculateExactTripleScore(reflist, candlist, alldict):
    newreflist = [[string.lower() for string in sublist] for sublist in reflist]
    newcandlist = [[string.lower() for string in sublist] for sublist in candlist]
    #First get all the classes by combining the triples in the candidatelist and referencelist
    allclasses = newcandlist + newreflist
    allclasses = [item for items in allclasses for item in items]
    allclasses = list(set(allclasses))

    lb = preprocessing.MultiLabelBinarizer(classes=allclasses)
    mcbin = lb.fit_transform(newcandlist)
    mrbin = lb.fit_transform(newreflist)

    precision = precision_score(mrbin, mcbin, average='macro')
    recall = recall_score(mrbin, mcbin, average='macro')
    f1 = f1_score(mrbin, mcbin, average='macro')

    alldict.update({'Exact_match': {'Precision': precision, 'Recall': recall, 'F1': f1}})

    return alldict

def evaluate(input_dataframe, outputfile_overall, outputfile_details):
    # allcand_ids, all_text, candlist, newcandlist = get_Cands_From_rebel_Tsv(candfile)
    # reflist, newreflist = getRefs(reffile, allcand_ids)
    # candlist, newcandlist = getCands(candfile)
    allcand_ids, all_text, allcandtriples, newcandlist, allreftriples, newreflist = get_Cands_and_Refs_from_csv(input_dataframe)
    starting_time = timeit.default_timer()
    print("Start time :",starting_time)
    totalsemevallist, totalsemevallistpertag = calculateAllScores(newreflist, newcandlist)
    function1_time = timeit.default_timer() 
    print("calculate all score time :", function1_time - starting_time)
    alldict, triple_score, combination_selected, triple_score_sum = calculateSystemScore(totalsemevallist, totalsemevallistpertag, newreflist, newcandlist)
    function2_time = timeit.default_timer() 
    print("calculate all score time :", function2_time - function1_time)
    alldict2 = calculateExactTripleScore(allreftriples, allcandtriples, alldict)
    with open(outputfile_overall, 'w') as outfile:
        json.dump(alldict2, outfile)

    all = {}
    # all['id'] = list(allcand_ids)
    all['id'] = allcand_ids.tolist()
    all['text'] = list(all_text)
    all['ref'] = allreftriples
    all['cand'] = allcandtriples
    all['triple_score'] = triple_score
    all['combination'] = combination_selected
    all['triple_score_sum'] = triple_score_sum
    # print(all)
    with open(outputfile_details, 'w') as outfile:
        json.dump(all, outfile)

def main():
    # Main function from benchmark.py
    df = benchmark()

    output_path = 'results/evaluation/llama/vicuna-7b-with-explanasion-test-combined.json'
    output_details_path = 'results/evaluation/llama/vicuna-7b-with-explanasion-test-combined-details.json'
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
