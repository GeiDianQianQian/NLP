# -*- coding: utf-8 -*-
import sys, codecs, optparse, os, math
from Queue import PriorityQueue, Queue

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
    def __init__(self, word, start_pos, log_prob):
        self.word        = word
        self.start_pos   = start_pos
        self.probability = log_prob

# the default segmenter does not use any probabilities, but you could ...
Pw     = Pdist(opts.counts1w)
# Pw2   = Pdist(opts.counts2w)
keys   = Pw.keys()
pq     = PriorityQueue()
queue  = Queue()
minint = -sys.maxint

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

def isNumber(x, beggining):
    # if is between 0 and 9 or has a separator
    return ((x >= u'\uFF10' and x <= u'\uFF19')
            or (not beggining and x == "·".decode('utf-8')))

def isPunctuation(x):
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

    if idx < len(input) and not isPunctuation(input[idx]):
        word += input[idx]
        idx += 1

    return (idx - 1, word)

def getEndWord(input, idx):
    i = idx
    while ((i < len(input)) and
          (not isNumber(input[i], True)) and
          (not isPunctuation(input[i]))):
        i += 1

    return (i - 1)

def getEntryMemoized(input, idx, endidx, cache, stack):
    # we will check up to 10 symbols at time, no more than that
    cnt  = 0
    word = ""
    endi = getEndWord(input, idx) if endidx == minint else endidx
    i    = endi
    arr  = Queue()
    # we start from thew
    while (i >= idx):
          word = input[i] + word
          # memoize values
          ans   = 0
          if word in cache:
              ans = cache[word]
          elif word in Pw:
              ans = Pw(word)
              cache[word] = ans
              entry = Entry(word, i, math.log(ans, 10))
              #blah = " pos: " + str(i) + " prob: " + str(math.log(ans, 10))
              #print("word: " + entry.word + blah)
              (_, stk) = getEntryMemoized(input, idx, i - 1, cache, stack)
              stack = stk
              arr.put(entry)
          else:
              cache[word] = None

          i -= 1

    prob_item = Entry("", 0, minint)
    while not arr.empty():
        item = arr.get()
        if item.probability > prob_item.probability:
            prob_item = item
    stack.append(prob_item)

    return (endi, stack)

right_p = -0.000001
for line in open(opts.input).readlines():
    utf8line = unicode(line.strip(), 'utf-8')

    idx = 0
    while idx < len(utf8line):
        ch = utf8line[idx]
        if isPunctuation(ch):
            entry = Entry(ch, idx, right_p)
            queue.put(entry)
        elif isNumber(ch, True):
            (i, number) = getNumber(utf8line, idx)
            entry = Entry(number, idx, right_p)
            idx = i
            queue.put(entry)
        else:
            (i, stk) = getEntryMemoized(utf8line, idx, minint, {}, [])
            for item in stk:
                if item.word != "":
                    queue.put(item)
            idx = i

        idx += 1

    while (not queue.empty()):
        item = queue.get()
        sys.stdout.write(item.word + " ")
    print("")

sys.stdout = old
