"""
A hashtable represented as a list of lists with open hashing.
Each bucket is a list of (key,value) tuples
"""

def htable(nbuckets):
    """Return a list of nbuckets lists"""
    htable = [[] for i in range(nbuckets)]
    return htable

def hashcode(o):
    """
    Return a hashcode for strings and integers; all others return None
    For integers, just return the integer value.
    For strings, perform operation h = h*31 + ord(c) for all characters in the string
    """
    if (type(o) == str or type(o) == int):
        if (type(o) == int):
            return o
        h = 0
        for c in o:
            h = h * 31 + ord(c)
        return h
    return None


def bucket_indexof(table, key):
    """
    You don't have to implement this, but I found it to be a handy function.
    Return the element within a specific bucket; the bucket is table[key].
    You have to linearly search the bucket.
    """



def htable_put(table, key, value):
    """
    Perform table[key] = value
    Find the appropriate bucket indicated by key and then append value to the bucket.
    If the bucket for key already has a key,value pair with that key then replace it.
    Make sure that you are only adding (key,value) associations to the buckets.
    """
    hash_key = hashcode(key)
    b_val = hash_key % len(table)
    bucket = table[b_val]
    getTable = htable_get(table, key)

    if getTable == None:
        bucket.append((key, [value]))
    else:
        for tup in bucket:
            if tup[0] == key:
                tup[1].append(value)

def htable_get(table, key):
    """
    Return table[key].
    Find the appropriate bucket indicated by the key and look for the association
    with the key. Return the value (not the key and not the association!)
    Return None if key not found.
    """
    hash_key = hashcode(key)
    b_val = hash_key % len(table)
    bucket = table[b_val] # bucket number

    for tup in bucket:
        if key == tup[0]:
            return tup[1]
    return None

def htable_buckets_str(table):
    """
    Return a string representing the various buckets of this table. The output looks like:
	0000->
	0001->
	0002->
	0003->parrt:99
	0004->
    where parrt:99 indicates an association of (parrt,99) in bucket 3.
    """
    reString = ''
    for i in range(0, len(table)):
        reString += "%04d->" % i
        listOfVals = []
        for tup in table[i]:
            listOfVals.append("%s:%s" %(tup[0], tup[1][0]))
        reString += (", ".join(listOfVals))+"\n"
    return reString

def htable_str(table):
    """
    Return what str(table) would return for a regular Python dict such as {parrt:99}.
    The order should be bucket order and then insertion order in bucket.
    The insertion order is guaranteed when you append to the buckets in htable_put().
    """
    reString = '{'
    listOfVals = []
    for i in range(0, len(table)):
        for tup in table[i]:
            listOfVals.append("%s:%s" %(tup[0], tup[1][0]))
    reString += ", ". join(listOfVals)
    reString += "}"
    return reString

