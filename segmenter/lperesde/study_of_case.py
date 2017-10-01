# -*- coding: utf-8 -*-

import sys, codecs, optparse, os, math
from Queue import PriorityQueue

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self: return float(self[key])/float(self.N)
        #else: return self.missingfn(key, self.N)
        elif len(key) == 1: return self.missingfn(key, self.N)
        else: return None

class Entry:
    # Entry to be used
    def __init__(self, word, start_pos, log_prob, size):
        self.word = word
        self.start_pos = start_pos
        self.size = size
        self.log_prob = log_prob

# the default segmenter does not use any probabilities, but you could ...
Pw  = Pdist(opts.counts1w)
keys= Pw.keys()
pq  = PriorityQueue()

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

def buildPossibilities(chart):
    #recursive
    while pq.empty() == False:
        # try to insert a word
        _,entry = pq.get()
        endindex = entry.start_pos + len(entry.word) - 1
        if(chart[endindex] != None):
            if(chart[endindex].log_prob < entry.log_prob):
                chart[endindex] = entry
            else:
                continue
        else:
            chart[endindex] = entry

        # gather candidates for next word
        next_start=endindex+1
        inserted = 0
        for allowed_length in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            next_word = input[next_start:next_start+allowed_length]
            if(next_word in Pw):
                next_entry = Entry(next_word,
                                   next_start,
                                   entry.log_prob + math.log(Pw(next_word), 2),
                                   entry)
                pq.put((next_start,next_entry))
                inserted += 1

        if((inserted == 0) and (next_start <= finalindex)):
            next_word = input[next_start]
            next_entry = Entry(next_word, next_start, entry.log_prob + math.log(Pw(next_word), 2), entry)
            pq.put((next_start, next_entry))

def isNumber(x, beggining):
    return ((x >= "０".decode('utf-8') and x <= "９".decode('utf-8'))
            or (not beggining and x == "·".decode('utf-8')))

def isPonctuation(x):
    return ((x >= u'\u3000' and x <= u'\u303F') or
           (x >= u'\uFF01' and x <= u'\uFF0F') or
           (x >= u'\uFF1A' and x <= u'\uFF20') or
           (x >= u'\uFF3B' and x <= u'\uFF40') or
           (x >= u'\uFF5B' and x <= u'\uFF65'))

def getNumber(input, idx):
    word = input[idx]
    idx += 1
    while idx < len(input) and isNumber(input[idx], False):
        word += input[idx]
        idx += 1

    if idx < len(input) and not isPonctuation(input[idx]):
        word += input[idx]
        idx += 1

    return (idx - 1, word)

count = 0
right_p = -0.000001
for line in open(opts.input).readlines():
    utf8line = unicode(line.strip(), 'utf-8')
    # take it reversed
    idx = 0
    while idx < len(utf8line):
        ch = utf8line[idx]
        entry = None
        if isPonctuation(ch):
            entry = Entry(ch, idx, right_p, None)
        elif isNumber(ch, True):
            (i, number) = getNumber(utf8line, idx)
            entry = Entry(number, idx, right_p , None)
            idx = i

        if (entry != None):
            pq.put((0, entry))
            print(entry.word)

        idx += 1
    # buildPossibilities(chart)
    #
    # wordlist = []
    # current_entry = chart[finalindex]
    # while(current_entry != None):
    #     wordlist.append(current_entry.word)
    #     current_entry = current_entry.back_ptr
    # wordlist.reverse()
    # print " ".join(wordlist)

sys.stdout = old
