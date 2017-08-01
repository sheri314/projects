from collections import defaultdict  # https://docs.python.org/2/library/collections.html

from words import get_text, words, filelist


def create_index(files):
    """
    Given a list of fully-qualified filenames, build an index from word
    to set of document indexes. The document index is just the index into the
    files parameter (indexed from 0).
    Make sure that you are mapping a word to a set, not a list.
    For each word w in file i, add i to the set of documents containing w
    Returns a dict object.
    """
    index = defaultdict(list)

    for i in range(0, len(files)):
        wordList = set(words(get_text(files[i])))
        for w in wordList:
            if i not in index[w]:
                index[w].append(i)

    return(index)

def index_search(files, index, terms):
    """
    Given an index and a list of fully-qualified filenames, return a list of them
    whose file contents has all words in terms as normalized by your words() function.
    Parameter terms is a list of strings.
    You can only use the index to find matching files; you cannot open the files and look inside.
    """
    listOfFiles = [] # give back the index numbers of the files that match
    for term in terms:
        listOfFiles.append(index[term])

    intersection = set(listOfFiles[0])
    for l in listOfFiles:
        intersection = intersection & set(l)

    result = [] # finds the intersection of all search terms
    for ind in list(intersection):
        result.append(files[ind])

    return(result)