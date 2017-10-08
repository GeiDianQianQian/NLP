# -*- coding: utf-8 -*-
import sys, codecs, optparse, os, math
from Queue import PriorityQueue

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-d", "--uniwsegcounts", dest='countswseg', default=os.path.join('wseg_data', 'count_wseg.txt'), help="unigram wseg counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None, V=0):
        self.voc=dict()
        self.maxlen = 0
        self.V=0
        for line in file(filename):
            (key, freq) = line.split(sep)
            if(" " in key):
                (before,after)=key.split(" ")
            else:
                before=key
            try:
                utf8key = unicode(key, 'utf-8')
                utf8before = unicode(before, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.voc[utf8before]=self.voc.get(utf8before, 0) + 1
            self.maxlen = max(len(utf8key), self.maxlen)
        self.V= float(len(self.keys()))
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: math.log(1,2)-math.log(float((N+self.V+1)),2))

    def __call__(self, key):
        if key in self:
            return math.log(float(self[key]+1),2)-math.log(float(self.N+self.V+1),2)
        else:
            return self.missingfn(key, self.N)

class Entry:
    #Entry to be used

    def __init__(self,word,start_pos,log_prob,back_ptr):
        self.word=word
        self.start_pos=start_pos
        self.log_prob=log_prob
        self.back_ptr=back_ptr

# the default segmenter does not use any probabilities, but you could ...
Pw  = Pdist(opts.counts1w)
Pwseg = Pdist(opts.countswseg)
Pw2  = Pdist(opts.counts2w)
pq  = PriorityQueue()

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

# ignoring the dictionary provided in opts.counts
input=""
count=0

def isNumber(x):
    idx = 0
    if (idx < len(x) and x[idx] >= u'\uFF10' and x[idx] <= u'\uFF19'):
        idx += 1
        while (idx < len(x) and ((x[idx] >= u'\uFF10' and x[idx] <= u'\uFF19')
            or (x[idx] == "·".decode('utf-8')))):
            idx += 1

        return idx == len(x)
    # if is between 0 and 9 or has a separator
    return False

with open(opts.input) as f:

    #repeat for each line of input
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        input=utf8line

        #initializing chart
        finalindex=len(input)-1
        chart = [None] * (finalindex+1)

        #initializing priorityqueue, gather candidates for first word
        inserted=0
        for allowed_length in range(1,Pw.maxlen):
            first_word=input[0:allowed_length]
            if (isNumber(first_word)):
                numberPw = -5.00000 / len(first_word);
                entry=Entry(first_word,0, numberPw,None)
                pq.put((0,entry))
                inserted+=1
            elif(first_word in Pw):
                entry=Entry(first_word,0,Pw(first_word),None)
                pq.put((0,entry))
                inserted+=1
            elif(first_word in Pwseg):
                # wseg is 3 times less reliable
                entry=Entry(first_word,0,Pwseg(first_word) * 3,None)
                pq.put((0,entry))
                inserted+=1
        if(inserted==0):
            first_word=input[0]
            entry = Entry(first_word,0,Pw(first_word), None)
            pq.put((0,entry))

        #recursive
        while pq.empty()==False:

            #try to insert a word
            _,entry=pq.get()
            endindex=entry.start_pos+len(entry.word)-1
            if(chart[endindex]!=None):
                if(chart[endindex].log_prob<entry.log_prob):
                    chart[endindex]=entry
                else:
                    continue
            else:
                chart[endindex] = entry

            #gather candidates for next word
            prev_word=entry.word
            next_start=endindex+1
            if(prev_word in Pw):
                inserted = 0
                for allowed_length in range(1,Pw2.maxlen,1):
                    next_word = input[next_start:next_start+allowed_length]
                    bigram=prev_word+" "+next_word
                    if (bigram in Pw2):
                        a=math.log((Pw2[bigram]+1), 2)
                        b1=Pw[prev_word]
                        b2=int(Pw2.voc.get(prev_word,0))
                        b=math.log((b1+b2+1),2)
                        logprob = a-b
                        next_entry = Entry(next_word, next_start, entry.log_prob+logprob, entry)
                        pq.put((next_start, next_entry))
                        inserted += 1
                    elif(next_word in Pw):
                        a=math.log(1, 2)
                        b=math.log((Pw[next_word]+1),2)
                        c=math.log(Pw.N+Pw.V+1)
                        d1=Pw[prev_word]
                        d2=int(Pw2.voc.get(prev_word,0))
                        d=math.log((d1+d2+1),2)
                        logprob = a+b-c-d
                        next_entry = Entry(next_word, next_start, entry.log_prob + logprob, entry)
                        pq.put((next_start, next_entry))
                        inserted +=1
                    else:
                        continue
                if ((inserted == 0) and (next_start <= finalindex)):
                    next_word = input[next_start]
                    a=math.log(1, 2)
                    b=math.log(1,2)
                    c=math.log(Pw.N+Pw.V+1)
                    d1=Pw[prev_word]
                    d2=int(Pw.voc.get(prev_word,0))
                    d=math.log((d1+d2+1),2)
                    logprob = a+b-c-d
                    next_entry = Entry(next_word, next_start, entry.log_prob+logprob, entry)
                    pq.put((next_start, next_entry))

            else:
                inserted = 0
                for allowed_length in range(1,Pw.maxlen,1):
                    next_word = input[next_start:next_start+allowed_length]
                    if(next_word in Pw):
                        next_entry = Entry(next_word, next_start, entry.log_prob+Pw(next_word), entry)
                        pq.put((next_start,next_entry))
                        inserted+=1
                if((inserted==0) and (next_start<=finalindex)):
                    next_word=input[next_start]
                    next_entry = Entry(next_word, next_start, entry.log_prob + Pw(next_word), entry)
                    pq.put((next_start, next_entry))

        wordlist=[]
        current_entry=chart[finalindex]
        while(current_entry!=None):
            wordlist.append(current_entry.word)
            current_entry=current_entry.back_ptr
        wordlist.reverse()
        print " ".join(wordlist)
sys.stdout = old
