# import regex as re

# dat = "joseph dorville `` joe '' walter -lrb- 16 august 1895 -- 23 may 1995 -rrb- was a professional footballer , who played for bristol rovers , huddersfield town , taunton united and bath city . he was the last surviving player to play under herbert chapman while at huddersfield . walter 's early footballing experience came while serving in the gloucestershire regiment during the first world war , when he represented the third battalion of the `` glorious glosters '' , before joining horfield united in the horfield area of his home city of bristol . after a year playing in the bristol and suburban association football league , he joined his first professional team in 1919 when he signed for bristol rovers . later in life he worked as a groundsman , firstly for the bristol co-operative society , before taking responsibility for bristol city 's ashton gate pitch in 1955 . he was later appointed as an assistant coach at his former club , bristol rovers , in 1960 when he was aged 65 . in huddersfield 's final game at leeds road , against blackpool on 30 april 1994 , he was the guest of honour . he was also present for town 's visit to bristol rovers ' twerton park . on 5 november 1994 , he made english football history as he became the oldest living professional player at 99 years , on the death of former brighton & hove albion player zillwood march in his 102nd year . "
# dat = dat.strip().replace(' ,', ',').replace(' .', '.').replace(' -lrb- ', ' (').replace(' -rrb- ', ') ')
# dat = dat.replace('``', '"').replace('\'\'', '"').replace(' \'', '\'')
# dat = re.sub(r'" (.*?) "', '"\g<1>"', dat)
# print(dat)
from datasets import load_dataset
import pandas as pd

dat = pd.DataFrame(load_dataset('UofA-LINGO/wikipedia-summaries-gpt-triples')['train'])
print(dat)