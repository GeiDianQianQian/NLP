#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref', help='input file (default data/hyp1-hyp2-ref)')
parser.add_argument('-n', '--num_sentences', default=None, type=int, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-b', '--beta', default=3.0, type=float, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-g', '--gamma', default=0.5, type=float, help='Number of hypothesis pairs to evaluate')
opts = parser.parse_args()


def word_matches(h, e):
    r = 0.0
    p = 0.0
    m = sum(1 for w in h if w in e) + 0.0001
    r = float(m)/float(len(e))
    p = float(m)/float(len(h))
    return p, r, m


def sentences():
    with open(opts.input) as f:
        for pair in f:
            yield [sentence.strip().split() for sentence in pair.split(' ||| ')]


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


for h1, h2, e in islice(sentences(), opts.num_sentences):
    p1, r1, m1 = word_matches(h1, e)
    p2, r2, m2 = word_matches(h2, e)
    c1 = get_c(h1, e)
    c2 = get_c(h2, e)
    l1 = (1-opts.gamma*(c1/m1)**opts.beta)*p1*r1/((1-opts.alpha)*r1+opts.alpha*p1)
    l2 = (1-opts.gamma*(c2/m2)**opts.beta)*p2*r2/((1-opts.alpha)*r2+opts.alpha*p2)
    if l1 > l2:
        print 1
    elif l1 == l2:
        print 0
    else:
        print -1



