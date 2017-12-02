#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import numpy as np
from collections import Counter

parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
parser.add_argument('-i', '--input', default='../data/hyp1-hyp2-ref', help='input file (default data/hyp1-hyp2-ref)')
parser.add_argument('-n', '--num_sentences', default=None, type=int, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-b', '--beta', default=3.0, type=float, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-g', '--gamma', default=0.5, type=float, help='Number of hypothesis pairs to evaluate')
opts = parser.parse_args()

def matches(h, e):
    r = 0.0
    p = 0.0
    m = sum(1 for w in h if w in e) + 0.0001
    r = float(m)/float(len(e))
    p = float(m)/float(len(h))
    f = 2 * p * r / (p + r)
    return p, r, f

def sentences():
    with open(opts.input) as f:
        for pair in f:
            yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

def get_ngrams(sentence, ref1, ref2, vc1, vc2):
    for n in xrange(1,5):
        e_ngrams  = [tuple(sentence[i:i+n]) for i in xrange(len(sentence)+1-n)]
        h1_ngrams = [tuple(ref1[i:i+n]) for i in xrange(len(ref1)+1-n)]
        h2_ngrams = [tuple(ref2[i:i+n]) for i in xrange(len(ref2)+1-n)]

        (vc1[n-1], vc1[n+3], vc1[n+7]) = matches(h1_ngrams, e_ngrams)
        (vc2[n-1], vc2[n+3], vc2[n+7]) = matches(h2_ngrams, e_ngrams)

    return (vc1, vc2)

for h1, h2, e in islice(sentences(), opts.num_sentences):
    vc1 = [None] * 32 # feature vector h1
    vc2 = [None] * 32 # feature vector h2
    (vc1, vc2) = get_ngrams(e, h1, h2, vc1, vc2)

    l1 = 0
    l2 = 0
    print l1
    if l1 > l2:
        print 1
    elif l1 == l2:
        print 0
    else:
        print -1



