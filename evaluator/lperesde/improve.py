#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import numpy as np
from HTMLParser import HTMLParser

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
    r = float(m)/float(len(e)) if e else 0.0001
    p = float(m)/float(len(h)) if h else 0.0001 
    f = 2 * p * r / (p + r)
    return p, r, f, m

def sentences():
    with open(opts.input) as f:
        for pair in f:
            yield [sentence.lower().strip().split() for sentence in pair.split(' ||| ')]

def get_ngrams(sentence, ref1, ref2, vc1, vc2):
    for n in xrange(1,5):
        e_ngrams  = [tuple(sentence[i:i+n]) for i in xrange(len(sentence)+1-n)]
        h1_ngrams = [tuple(ref1[i:i+n]) for i in xrange(len(ref1)+1-n)]
        h2_ngrams = [tuple(ref2[i:i+n]) for i in xrange(len(ref2)+1-n)]

        # save precison, score and f1 for ngrams
        (vc1[n-1], vc1[n+3], vc1[n+7], _) = matches(h1_ngrams, e_ngrams)
        (vc2[n-1], vc2[n+3], vc2[n+7], _) = matches(h2_ngrams, e_ngrams)

    # average of ngrams
    vc1[12] = (vc1[0]+vc1[1]+vc1[2]+vc1[3])/4
    vc2[12] = (vc2[0]+vc2[1]+vc2[2]+vc2[3])/4

    return (vc1, vc2)

def get_pos(sentence, ref1, ref2, vc1, vc2):
    # no-op
    return (vc1, vc2)

def lcs(s1, s2):
   m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
   longest, x_longest = 0, 0
   for x in xrange(1, 1 + len(s1)):
       for y in xrange(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
   return s1[x_longest - longest: x_longest]

def get_c(s1, s2):
    mylist=[]
    while lcs(s2,s1) != []:
        x = lcs(s2, s1)
        mylist.append(x)
        s1 = [i for i in s1 if i not in x]
        s2 = [i for i in s2 if i not in x]
    return len(mylist)

def get_words(sentence, ref1, ref2):
    p1, r1, _, m1 = matches(ref1, sentence)
    p2, r2, _, m2 = matches(ref2, sentence)
    c1 = get_c(ref1, sentence)
    c2 = get_c(ref2, sentence)
    l1 = (1-opts.gamma*(c1/m1)**opts.beta)*p1*r1/((1-opts.alpha)*r1+opts.alpha*p1)
    l2 = (1-opts.gamma*(c2/m2)**opts.beta)*p2*r2/((1-opts.alpha)*r2+opts.alpha*p2)
    
    return (l1, l2)

def fix_input(h):
    newh = [w.replace("&quot;", '"') for w in h]
    return newh

for h1, h2, e in islice(sentences(), opts.num_sentences):
    vc1 = [0] * 32 # feature vector h1
    vc2 = [0] * 32 # feature vector h2
    h1 = fix_input(h1)
    h2 = fix_input(h2)
    (vc1, vc2) = get_ngrams(e, h1, h2, vc1, vc2)
    (vc1, vc2) = get_pos(e, h1, h2, vc1, vc2)
    l1 = sum(vc1)
    l2 = sum(vc2)
    if l1 == l2:
        print 0
    elif l1 > l2:
        print 1
    else:
        print -1


