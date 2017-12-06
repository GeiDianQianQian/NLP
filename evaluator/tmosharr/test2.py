import math
import nltk
from collections import Counter
import argparse  # optparse is deprecated
from itertools import islice  # slicing for iterators
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
    r = float(m) / float(len(e))
    p = float(m) / float(len(h))
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
    mylist = []
    while lcs(s2, s1) != []:
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

for h1, h2, e in islice(sentences(), opts.num_sentences):
    print nltk.pos_tag(h1)

