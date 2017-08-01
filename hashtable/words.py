import os
import re
import string


def filelist(root):
    """Return a fully-qualified list of filenames under root directory"""
    f = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            f.append(os.path.join(path, name))
    return f

def get_text(fileName):
    f = open(fileName)
    s = f.read()
    f.close()
    return s


def words(text):
    """
    Given a string, return a list of words normalized as follows.
    Split the string to make words first by using regex compile() function
    then string.punctuation + '0-9\\r\\t\\n]' to replace all those
    char with a space character.
    Split on space to get word list.
    Ignore words < 3 char long.
    Lowercase all words
    """
    pattern = re.compile("[" + string.punctuation + "0-9\\r\\t\\n]")
    string1 = ''
    for t in text:
        string1+=(re.sub(pattern, ' ', t))
    string2 = string1.split(' ')

    newStrings = set()
    for word in string2:
        if len(word) >= 3 and word not in newStrings:
            newStrings.add(word)
    wordList = [word.lower() for word in newStrings]

    return wordList


def results(docs, terms):
    """
    Given a list of fully-qualifed filenames, return an HTML file
    that displays the results and up to 2 lines from the file.
    Return at most 100 results.  Arg terms is a list of string terms.
    """
    template = """
    <html>
    <body>
    <h2> Search results for %s in %s files! </h2>
    %s
    </body>
    </html>
    """
    sum = 0
    stringOfFiles = ""

    for file in docs:
        stringOfFiles = stringOfFiles + "<a href = file://" + file + ">" + file + "</a><br>"
        f = open(file, "r")
        contents = f.read()
        f.close()
        lines = contents.lstrip().split('\n')
        firstTwoLines = lines[3].lstrip()+ '<br>' + lines[4].lstrip() + '<br><br><br>'
        stringOfFiles += firstTwoLines
        sum +=1
        if sum == 100:
            break

    sum = str(sum)

    headerTerms = ''
    for x in terms:
        headerTerms += x + " "
    html = template % (headerTerms, sum, stringOfFiles)
    return html

def filenames(docs):
    """Return just the filenames from list of fully-qualified filenames"""
    if docs is None:
        return []
    return [os.path.basename(d) for d in docs]
