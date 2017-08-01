import language_check
import pandas as pd
import sys

from multiprocessing import Pool

def language_mistake_count(text):
    # Counts the number of language mistakes including spelling and grammar mistakes
    tool = language_check.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)

text_df = pd.read_csv('train.tsv', delimiter='\t', header=0)
data_list = text_df['EssayText'][0:5].tolist()

def creat_print(i):
    result = language_mistake_count(i)
    print "{0}:{1}".format(data_list.index(i), result)
    # print "{0}:{1}".format(i,result)   #debug

if __name__ == '__main__':

## M cores
  pool = Pool()
  pool.map(creat_print, data_list)
  pool.close()
  pool.join()
