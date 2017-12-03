import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import string
import math

parser = argparse.ArgumentParser(description='Get ngrams.')
parser.add_argument('-i', '--input', default='../data/hyp1-hyp2-ref', help='input file (default data/hyp1-hyp2-ref)')
parser.add_argument('-n', '--num_sentences', default=None, type=int, help='Number of hypothesis pairs to evaluate')
opts = parser.parse_args()

def sentences():
    with open(opts.input) as f:
        for pair in f:
            yield [sentence.translate(None, string.punctuation).lower().strip().split() for sentence in pair.split(' ||| ')]

dictionary = [{} for _ in xrange(1,5)]
for _, _, e in islice(sentences(), opts.num_sentences):
    e = ['<s>'] + e + ['</s>']
    for n in xrange(1,5):
        for i in xrange(len(e)+1-n):
            tp = tuple(e[i:i+n])
            if tp in dictionary[n-1]:
                dictionary[n-1][tp] += 1.0
            else:
                dictionary[n-1][tp] = 1.0

for n in xrange(1,5):
    for key in dictionary[n-1].keys():
        if n == 1: #unigram
            print ' '.join(key) + ' ||| ' + str(math.log10(dictionary[n-1][key]/len(dictionary[n-1])))
        elif n == 2: #bigram
            print ' '.join(key) + ' ||| ' + str(math.log10(dictionary[n-1][key]/dictionary[n-2][key[:1]]))
        elif n == 3: #trigram
            print ' '.join(key) + ' ||| ' + str(math.log10(dictionary[n-1][key]/dictionary[n-2][key[:2]]))
        else: #4-gram
            print ' '.join(key) + ' ||| ' + str(math.log10(dictionary[n-1][key]/dictionary[n-2][key[:3]]))

