#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import numpy as np
import math
from collections import Counter
import argparse  # optparse is deprecated
from itertools import islice  # slicing for iterators
import numpy as np
from nltk.corpus import stopwords
import string

parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref', help='input file (default data/hyp1-hyp2-ref)')
parser.add_argument('-n', '--num_sentences', default=None, type=int, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-a', '--alpha', default=0, type=float, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-b', '--beta', default=4.0, type=float, help='Number of hypothesis pairs to evaluate')
parser.add_argument('-g', '--gamma', default=0.5, type=float, help='Number of hypothesis pairs to evaluate')
opts = parser.parse_args()


def word_matches(h, e):
    r = 0.0
    p = 0.0
    m = sum(1 for w in h if w in e) + 0.0001
    r = float(m)/(float(len(e))+0.0001)
    p = float(m)/(float(len(h))+0.0001)
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


# Collect BLEU-relevant statistics for a single sentence/reference pair.
# Return value is a generator yielding:
# (c, r, numerator1, denominator1, ... numerator4, denominator4)
# Summing the columns across calls to this function on an entire corpus will
# produce a vector of statistics that can be used to compute BLEU (below)
def bleu_stats(sentence, reference):
    stats = []
    stats.append(len(sentence))
    stats.append(len(reference))
    for n in xrange(1, 5):
        s_ngrams = Counter([tuple(sentence[i:i + n]) for i in xrange(len(sentence) + 1 - n)])
        r_ngrams = Counter([tuple(reference[i:i + n]) for i in xrange(len(reference) + 1 - n)])
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(sentence) + 1 - n, 0]))
    return stats


# Compute BLEU from collected statistics obtained by call(s) to bleu_stats
def bleu(stats):
    if len(filter(lambda x: x == 0, stats)) > 0:
        return 0
    (c, r) = stats[:2]
    bleu_prec = sum([math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])])
    return math.exp(min([0, 1 - float(r) / c]) + 0.25 * bleu_prec)


# A modification of BLEU that returns a positive value even when some
# higher-order precisions are zero. From Liang et al. 2006 (Footnote 5):
# http://aclweb.org/anthology-new/P/P06/P06-1096.pdf
def smoothed_bleu(stats):
    return sum([bleu(stats[:2 + 2 * i]) / math.pow(2, 4 - i + 1) for i in xrange(1, 5)])


def bl(h1, h2, e):
    stats1 = bleu_stats(h1, e)
    stats2 = bleu_stats(h2, e)
    score1 = smoothed_bleu(stats1)
    score2 = smoothed_bleu(stats2)
    if score1 > score2:
        return 1
    elif score1 == score2:
        return 0
    else:
        return -1


def mateor(h1, h2, e):
    p1, r1, m1 = word_matches(h1, e)
    p2, r2, m2 = word_matches(h2, e)
    c1 = get_c(h1, e)
    c2 = get_c(h2, e)
    l1 = (1-opts.gamma*(c1/m1)**opts.beta)*p1*r1/((1-opts.alpha)*r1+opts.alpha*p1)
    l2 = (1-opts.gamma*(c2/m2)**opts.beta)*p2*r2/((1-opts.alpha)*r2+opts.alpha*p2)
    if l1 > l2:
        return 1
    elif l1 == l2:
        return bl(h1, h2, e)
    else:
        return -1



#!/usr/bin/env python

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

def get_ngrams(sentence, ref1, ref2, vc1, vc2):
    for n in xrange(1,5):
        e_ngrams  = [tuple(sentence[i:i+n]) for i in xrange(len(sentence)+1-n)]
        h1_ngrams = [tuple(ref1[i:i+n]) for i in xrange(len(ref1)+1-n)]
        h2_ngrams = [tuple(ref2[i:i+n]) for i in xrange(len(ref2)+1-n)]

        # save precison, score and f1 for ngrams
        (vc1[n-1], vc1[n+3], vc1[n+7]) = matches(h1_ngrams, e_ngrams)
        (vc2[n-1], vc2[n+3], vc2[n+7]) = matches(h2_ngrams, e_ngrams)

    # average of ngrams
    vc1[12] = (vc1[0]+vc1[1]+vc1[2]+vc1[3])/4
    vc2[12] = (vc2[0]+vc2[1]+vc2[2]+vc2[3])/4

    return (vc1, vc2)

def get_pos(sentence, ref1, ref2, vc1, vc2):
    # no-op
    return (vc1, vc2)

def fix_input(h):
    h = [w.replace("&quot;", '"') for w in h]
    return h

# remove stop words
def rsw(h):
    return [word for word in h if word not in cachedStopWords]

cachedStopWords = stopwords.words("english")
for h1, h2, e in islice(sentences(), opts.num_sentences):
    vc1, vc2  = [0] * 32, [0] * 32 # feature vector h1
    sw1, sw2 = [0] * 13, [0] * 13
    h1 = fix_input(h1)
    h2 = fix_input(h2)
    (vc1, vc2) = get_ngrams(e, h1, h2, vc1, vc2)
    (sw1, sw2) = get_ngrams(rsw(e), rsw(h1), rsw(h2), sw1, sw2)
    l1 = (sum(vc1) + (sum(sw1) * 1.1))/2.1
    l2 = (sum(vc2) + (sum(sw2) * 1.1))/2.1
    if l1 == l2:
        print mateor(h1, h2, e)
    elif l1 > l2:
        print 1
    else:
        print -1



