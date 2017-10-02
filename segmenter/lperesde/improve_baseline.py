# -*- coding: utf-8 -*-
import sys, codecs, optparse, os, math
from Queue import Queue

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

    def probOfUnknown(self):
        return 0.5/float(self.N)

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

def getEndWord(input, idx, sub):
    i = idx
    while ((i < len(input)) and
          (not isNumber(input[i], True)) and
          (not isPunctuation(input[i]))):
        i += 1

    return (i - 1 - sub)

# get biggest words
def getGoodStack(item, stack):
    if not stack:
        if not item:
            return []
        else:
            stack.append(item)
            return stack

    if (item == None):
        item = stack.pop()
        stack = getGoodStack(item, stack)
    else:
        #if len(stack) >= len(item.word)-1:
        # slice last elements
        nslice = (len(item.word) - 1) * (-1);
        items = stack[nslice:]
        if nslice != 0:
            for _ in range(len(items)): stack.pop()
            if (items[0].start_pos == item.start_pos):
                stack.append(item)
            else:
                # if not, put them back
                stack.extend(items)
                stack.append(item)
        else:
            # if we did not do anything, append item back
            stack.append(item)

    return stack

def getUnknown(utf8line, idx, sub):
    if (sub >= len(utf8line) - idx):
        return (0, utf8line)
    else:
        return (idx, utf8line[idx : idx + sub])

def getEntryMemoized(input, idx, endidx, sub, cache, stack):
    # we will check up to 10 symbols at time, no more than that
    cnt  = 0
    word = ""
    endi = getEndWord(input, idx, sub) if endidx == minint else endidx
    i    = endi
    arr  = Queue()
    # we start from thew
    while (i >= idx):
          word = input[i] + word
          # memoize values
          ans   = 0
          entry = None
          if word in cache:
              ans = cache[word]
          elif word in Pw:
              ans = Pw(word) * 5 if len(word) == 2 else Pw(word)
              cache[word] = ans
              entry = Entry(word, i, math.log(ans, 10))
            #   blah = " pos: " + str(i) + " prob: " + str(math.log(ans, 10))
            #   print("word: " + entry.word + blah)
              (_, stk) = getEntryMemoized(input, idx, i - 1, sub, cache, stack)
              stack = stk
              arr.put(entry)
          else:
              ans = Pw.probOfUnknown()
              #entry = Entry(word, i, math.log(ans, 10))
              cache[word] = ans
              #arr.put(entry)

          i -= 1

    prob_item = Entry("", 0, minint)
    while not arr.empty():
        item = arr.get()
        if item.probability > prob_item.probability:
            prob_item = item

    if (prob_item.word != ""):
        stack.append(prob_item)

    stack = getGoodStack(None, stack)
    # print ([i.word for i in stack])

    return (endi, stack)

right_p = -0.000001
cnt = 0
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
            if len(utf8line) > 0:
                sub = 0
                stk = []
                oldidx = idx
                while (sub < len(utf8line) - oldidx):
                    (i, stk) = getEntryMemoized(utf8line, idx, minint, sub, {}, [])
                    if not stk:
                        sub += 1
                    else:
                        idx = i
                        break;

                if sub > 0:
                    (i, unknown) = getUnknown(utf8line, idx + 1, sub)
                    stk.append(Entry(unknown, i, right_p))
                    idx = i + len(unknown) -1

                for item in stk:
                    queue.put(item)

        idx += 1
    cnt += 1

    # print("======================== INPUT ")
    # print(utf8line)
    # print("======================== OUTPUT ")
    if (not queue.empty()):
        item = queue.get()
        sys.stdout.write(item.word)
    while (not queue.empty()):
        item = queue.get()
        sys.stdout.write(" " + item.word)
    print("")
    # print("========================")
    #
    # if (cnt == 3):
    #     exit(0)

sys.stdout = old
