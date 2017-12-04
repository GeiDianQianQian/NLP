#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import numpy as np
from nltk.corpus import stopwords
import string

parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
parser.add_argument('-i', '--input', default='../data/hyp1-hyp2-ref', help='input file (default data/hyp1-hyp2-ref)')
parser.add_argument('-m', '--model', default='./model/ngram_model', help='input file (model)')
parser.add_argument('-n', '--num_sentences', default=None, type=int, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-b', '--beta', default=3.0, type=float, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-g', '--gamma', default=0.5, type=float, help='Number of hypothesis pairs to evaluate')
opts = parser.parse_args()

cachedStopWords = stopwords.words("english")
ngram_dict = {}

def matches(h, e):
    r = 0.0
    p = 0.0
    m = sum(1 for w in h if w in e) + 0.0001
    r = float(m)/float(len(e)) if e else 0.0001
    p = float(m)/float(len(h)) if h else 0.0001 
    f = 2 * p * r / (p + r)
    return p, r, f

def sentences():
    with open(opts.input) as f:
        for pair in f:
            yield [sentence.translate(None, string.punctuation).lower().strip().split() for sentence in pair.split(' ||| ')]

def get_model():
    with open(opts.model) as f:
        for pair in f:
            yield tuple(pair.split(' ||| '))

def score_ngram(ngram):
    if ngram in ngram_dict:
        return ngram_dict[ngram]
    else:
        return -10

def get_ngrams(reference, cand1, cand2, vc1, vc2, long):
    score_cand1 = 0
    score_cand2 = 0
    for n in xrange(1,5):
        e_ngrams  = [tuple(reference[i:i+n]) for i in xrange(len(reference)+1-n)]
        h1_ngrams = [tuple(cand1[i:i+n]) for i in xrange(len(cand1)+1-n)]
        h2_ngrams = [tuple(cand2[i:i+n]) for i in xrange(len(cand2)+1-n)]

        if long:
            for i in xrange(len(cand1)+1-n):
                score_cand1 += score_ngram(tuple(cand1[i:i+n]))
            for i in xrange(len(cand2)+1-n):
                score_cand2 += score_ngram(tuple(cand2[i:i+n]))

        # save precison, score and f1 for ngrams
        (vc1[n-1], vc1[n+3], vc1[n+7]) = matches(h1_ngrams, e_ngrams)
        (vc2[n-1], vc2[n+3], vc2[n+7]) = matches(h2_ngrams, e_ngrams)

    # average of ngrams
    vc1[12] = (vc1[0]+vc1[1]+vc1[2]+vc1[3])/4
    vc2[12] = (vc2[0]+vc2[1]+vc2[2]+vc2[3])/4

    if long:
        vc1[13] = score_cand1/100
        vc2[13] = score_cand2/100

    return (vc1, vc2)

def fix_input(h):
    h = [w.replace("&quot;", '"') for w in h]
    return h

# remove stop words
def rsw(h):
    return [word for word in h if word not in cachedStopWords]

for words, count in get_model():
    tp = tuple(words.split())
    ngram_dict[tp] = float(count)

for h1, h2, e in islice(sentences(), opts.num_sentences):
    vc1, vc2  = [0] * 32, [0] * 32 # feature vector h1
    sw1, sw2 = [0] * 13, [0] * 13
    h1 = fix_input(h1)
    h2 = fix_input(h2)
    (vc1, vc2) = get_ngrams(e, h1, h2, vc1, vc2, True) 
    (sw1, sw2) = get_ngrams(rsw(e), rsw(h1), rsw(h2), sw1, sw2, False)
    l1 = (sum(vc1[0:13]) + (sum(sw1) * 1.1) + (vc1[13]*0.4))/2.5
    l2 = (sum(vc2[0:13]) + (sum(sw2) * 1.1 + (vc1[13]*0.4)))/2.5
    if l1 == l2:
        print 0
    elif l1 > l2:
        print 1
    else:
        print -1


