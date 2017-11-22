#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
from collections import defaultdict
import itertools

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="../data/input",
                     help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="../data/tm",
                     help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="../data/lm",
                     help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int",
                     help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int",
                     help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=10)")
optparser.add_option("-d", "--distortion-limit", dest="d", default=5, type="int", help="Distortion limit (default=5)")
optparser.add_option("-y", "--distortion-penalty", dest="y", default=-0.01, type="float", help="Distortion penalty (default=0.01)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
                     help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

p_phrase = namedtuple("p_phrase", "start,end,english,logprob")

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)

'''
def get_list_of_phrases(phrase):
    english=phrase.english
    logprob=phrase.logprob
    phrases=[]
    words=english.split()
    if len(words)>opts.p:
        phrases.append(phrase)
        return phrases
    else:
        permutations=list(itertools.permutations(words, len(words)))
        for permutation in permutations:
            new_english=" ".join(permutation)
            new_phrase=models.phrase(new_english,logprob)
            phrases.append(new_phrase)
        return phrases
'''

def ph(h):
    s=set()
    for phrase_list in all_phrases:
        for p in all_phrases[phrase_list]:
            valid=True
            for idx in range(p.start, p.end+1):
                if h.bitmap[idx]==True:
                    valid=False
            if abs(h.end+1-p.start)>5:
                valid=False
            if valid==True:
                s.add(p)
    return s


def nxt(h,p):
    lst=p.english.split()
    last=lst[-1:]
    second_last=lst[-2:-1]
    st=(second_last,last)
    bits=h.bitmap
    for i in range(p.start,p.end+1):
        bits[i]=True
    r=p.end


def get_all_phrases(f):
    all_phrases = defaultdict(list)
    for i in range(len(f) - 1):
        for j in range(i + 1, len(f)):
            if f[i:j] in tm:
                elem_list = tm[f[i:j]]
                for elem in elem_list:
                    all_phrases[(i, j)].append(p_phrase(i, j, elem.english, elem.logprob))
    return all_phrases


def update_stacks(position, h, phrase, word_tuple, is_end):
    logprob = h.logprob + phrase.logprob
    lm_state = h.lm_state
    for word in phrase.english.split():
        (lm_state, word_logprob) = lm.score(lm_state, word)
        logprob += word_logprob
    if is_end:
        logprob += lm.end(lm_state)
    new_hypothesis = hypothesis(logprob, lm_state, h, phrase, word_tuple, 0, [False]*len(f))
    if lm_state not in stacks[position] or stacks[position][lm_state].logprob < logprob:
        stacks[position][lm_state] = new_hypothesis
    return stacks


def extract_english(h):
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)


def extract_english_phrases(h, arr, english):
    if h.predecessor is None:
        return arr
    else:
        arr.append(h.phrase.english) if english else arr.append(h.f)
        return extract_english_phrases(h.predecessor, arr, english)


def score(arr):
    logprob = 0.0
    auxArr = arr[:]
    auxArr.insert(0, lm.begin()[0])
    for i, word in enumerate(auxArr[:-1]):
        try:
            (lm_state, word_logprob) = lm.score(tuple(word.split()[-2:]), auxArr[i + 1].split()[0])
            logprob += word_logprob
        except:
            logprob += -6
    return logprob


def swap(arr, s):
    best = arr[:]
    bests = s
    changed = False
    for i in range(len(arr[:-2])):
        for j in xrange(i + 1, len(arr[:-1])):
            aux = arr[:]
            aux[i], aux[j] = aux[j], aux[i]
            sc = score(aux)
            if (sc > bests):
                bests = sc
                best = aux[:]
                changed = True
    return (best, bests, changed)


def improve(winner):
    current = extract_english_phrases(winner, [], True)[::-1]
    # currentF = extract_english_phrases(winner, [], False)[::-1]
    while True:
        s_current = score(current)
        (current, s_current, c1) = swap(current, s_current)
        if not c1:
            break
    return current


def extract_tm_logprob(h):
    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)


french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french, ())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))

for f in french:
    all_phrases = get_all_phrases(f)
    hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, f, end, bitmap")
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, None, 0, [False]*len(f))
    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
        for h in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:opts.s]:  # prune
            for j in xrange(i + 1, len(f) + 1):
                if f[i:j] in tm:
                    for phrase in tm[f[i:j]]:
                        stacks = update_stacks(j, h, phrase, f[i:j], j == len(f))

    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
    w = improve(winner)
    print " ".join(w)

    if opts.verbose:
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
                         (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
