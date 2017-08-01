# Got slate magazine data from http://www.anc.org/data/oanc/contents/
# rm'd .xml, .anc files, leaving just .txt
# 4534 files in like 55 subdirs

from htable import *
from words import get_text, words

def myhtable_create_index(files):
    """
    Build an index from word to set of document indexes
    This does the exact same thing as create_index() except that it uses
    your htable.  As a number of htable buckets, use 4011.
    Returns a list-of-buckets hashtable representation.
    """
    index = htable(4011)
    for i in range(0, len(files)):
        for w in words(get_text(files[i])):
            htable_put(index, w, i)

    return(index)

def myhtable_index_search(files, index, terms):
    """
    This does the exact same thing as index_search() except that it uses your htable.
    I.e., use htable_get(index, w) not index[w].
    """
    result = []
    if htable_get(index, terms[0]) != None:
        intersection = set(htable_get(index,terms[0]))
        if len(intersection) != 0:
            for t in terms:
                getVal = htable_get(index, t)
                intersection = set(getVal) & set(intersection)
            for i in range(0, len(files)):
                if i in intersection:
                    result.append(files[i])

    return result
