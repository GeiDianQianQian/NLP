
import sys, codecs, optparse, os, math
from Queue import PriorityQueue

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('../data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('../data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('../data', 'input'), help="input file to segment")
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
    #Entry to be used

    def __init__(self,word,start_pos,log_prob,back_ptr):
        self.word=word
        self.start_pos=start_pos
        self.log_prob=log_prob
        self.back_ptr=back_ptr

class MyPriorityQueue(PriorityQueue):
    def __init__(self):
        PriorityQueue.__init__(self)
        self.counter = 0

    def __len__(self):
        PriorityQueue.__len__(self)

    def put(self, item, priority):
        PriorityQueue.put(self, (priority, self.counter, item))
        self.counter += 1

    def get(self, *args, **kwargs):
        _, _, item = PriorityQueue.get(self, *args, **kwargs)
        return item


# the default segmenter does not use any probabilities, but you could ...
Pw  = Pdist(opts.counts1w)
keys=Pw.keys()
pq = MyPriorityQueue()

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

# ignoring the dictionary provided in opts.counts
input=""
count=0
with open(opts.input) as f:

    #getting the input ready
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        input+=utf8line

    finalindex=len(input)
    chart = [None] * finalindex

    #initializing priorityqueue
    for key in keys:
        if(key.startswith(input[0])):
            entry=Entry(key,0,math.log(Pw(key),2),None)
            pq.put(entry,0)

    #recursive
    while pq.empty()==False:
        entry=pq.get()
        endindex=entry.start_pos+len(entry.word)-1
        if(chart[endindex]!=None):
            if(chart[endindex].log_prob<entry.log_prob):
                chart[endindex]=entry
            else:
                continue
        else:
            chart[endindex] = entry
        new_start=endindex+1
        for new_word in keys:
            if (new_word.startswith(input[new_start])):
                new_entry = Entry(new_word, new_start, entry.log_prob+math.log(Pw(new_word), 2), entry)
                pq.put(new_entry, new_start)

    for el in chart:
        if el != None and el.word != None:
            print(el.word+" ")

sys.stdout = old
