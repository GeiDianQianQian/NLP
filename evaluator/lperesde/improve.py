#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wdn
from nltk.stem import wordnet as wn
from nltk import pos_tag
from nltk import word_tokenize
from itertools import chain
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
wnlemma = wn.WordNetLemmatizer()
ngram_dict = {}

def wn_contains(word, ref):
    # compare only unigrams
    if len(word) > 1:
        return False
    synonyms = wdn.synsets(''.join(word))
    synset   = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
    refset   = set([''.join(r) for r in ref])
    result   = bool(synset & refset)
    return result # check intersection of sets

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def is_similar(word, ref):
    # compare only unigrams
    if len(word) > 1:
        return False
    synonyms = [word.lemma_names() for word in wdn.synsets(''.join(word))]
    words = [''.join(r) for r in ref]
    for syn in chain.from_iterable(synonyms):
        for w in words:
            if levenshtein(str(syn), str(w)) > 0.8:
                return True
    return False

def matches(h, e):
    r = 0.0
    p = 0.0
    m = 0.0001
    for w in h:
        # 'wn_contains' is expensive, so it only goes there
        # in case that we did not find it (a second try)
        if w in e or wn_contains(w, e) or is_similar(w, e):
            m += 1
    r = float(m)/float(len(e)) if e else 0.0001
    p = float(m)/float(len(h)) if h else 0.0001 
    f = 2 * p * r / (p + r)
    return p, r, f

def get_type_wordnet(tag):
    ADJ, ADV, NOUN, VERB = 'a', 'r', 'n', 'v'
    if tag.startswith('N'):
        return NOUN
    elif tag.startswith('V'):
        return VERB
    elif tag.startswith('J'):
        return ADJ
    elif tag.startswith('R'):
        return ADV
    
    return VERB

def sentences():
    with open(opts.input) as f:
        for pair in f:
            value = [[],[],[]]
            for i,sentence in enumerate(pair.split(' ||| ')):
                sentence = sentence.decode('unicode_escape').encode('ascii','ignore').lower()
                arr = [wnlemma.lemmatize(''.join(w[:1]), get_type_wordnet(''.join(w[1:])))
                             for w in pos_tag(word_tokenize(sentence))]
                # remove punctuation
                value[i] = str(" ".join(arr)).translate(None, string.punctuation).strip().split()
            yield value

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
    l1 = (sum(vc1[0:8])*0.5 + (sum(vc1[8:13]) * 2) + (sum(sw1) * 1.1) + (vc1[13]*0.4))/4
    l2 = (sum(vc2[0:8])*0.5 + (sum(vc2[8:13]) * 2) + (sum(sw2) * 1.1) + (vc1[13]*0.4))/4
    if l1 == l2:
        print 0
    elif l1 > l2:
        print 1
    else:
        print -1
