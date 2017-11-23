#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input_small",
                     help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm",
                     help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm",
                     help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int",
                     help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int",
                     help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=10, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-d", "--distortion-limit", dest="d", default=4, type="int", help="Distortion limit (default=5)")
optparser.add_option("-y", "--distortion-penalty", dest="y", default=-2, type="float",
                     help="Distortion penalty (default=-1)")
optparser.add_option("-w", "--beam-width", dest="beam_width", default=0.5, type="float",
                     help="Beam width (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
                     help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
p_phrase = namedtuple("p_phrase", "start, end, english, logprob")
hypothesis = namedtuple("hypothesis", "lm_state, bitmap, end, logprob, predecessor, last_phrase")
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]


def extract_english_phrases(h, word_list, length):
    arr1 = [None] * length
    while (True):
        if h.last_phrase is None:
            break
        start = h.last_phrase.start
        english = h.last_phrase.english
        arr1[start] = english
        if h.predecessor is not None:
            h = h.predecessor
        else:
            break
    arr2 = []
    for phrase in arr1:
        if phrase is not None:
            arr2.append(phrase)
    return arr2


def sequence_score(arr):
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
            sc = sequence_score(aux)
            if (sc > bests):
                bests = sc
                best = aux[:]
                changed = True
    return (best, bests, changed)


def improve(current):
    while True:
        s_current = sequence_score(current)
        (current, s_current, c1) = swap(current, s_current)
        if not c1:
            break
    return current


def extract_tm_logprob(h):
    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)


def beam(stack):
    bests = sorted(stack.itervalues(), key=lambda h: -h.logprob)
    maxp = bests[0].logprob
    min_allowable = maxp - opts.beam_width
    bm = []
    for h in bests:
        if h.logprob >= min_allowable:
            bm.append(h)
    return bm


def get_hypothesis_length(h):
    count = 0
    for bit in h.bitmap:
        if bit:
            count += 1
    return count


def get_all_phrases(f, tm):
    all_p_phrases = defaultdict(list)
    length = len(f)
    for i in range(length):
        for j in range(i + 1, length + 1):
            sub_tuple = f[i:j]
            if sub_tuple in tm.keys():
                list_of_phrases = tm[sub_tuple]
                for phrase in list_of_phrases:
                    new_p_phrase = p_phrase(i, j - 1, phrase.english, phrase.logprob)
                    all_p_phrases[(i, j - 1)].append(new_p_phrase)
    return all_p_phrases


def ph(h, all_p_phrases):
    s = []
    keys = all_p_phrases.keys()
    for key in keys:
        p_phrase_list = all_p_phrases[key]
        for p_phrase in p_phrase_list:
            start = p_phrase.start
            end = p_phrase.end
            valid = True
            for i in range(start, end + 1):
                if h.bitmap[i]:
                    valid = False
                    if abs(h.end + 1 - start) > opts.d:
                        valid = False
            if valid:
                s.append(p_phrase)
    return s


def get_next_hypothesis(h, p, lm):
    english = p.english
    logprob = h.logprob + p.logprob
    start = p.start
    end = p.end
    lm_state = h.lm_state
    for english_word in english.split():
        (lm_state, english_word_logprob) = lm.score(lm_state, english_word)
        logprob += english_word_logprob
    #logprob += lm.end(lm_state)
    logprob += opts.y * abs(h.end + 1 - start)
    bits = h.bitmap[:]
    for i in range(start, end + 1):
        bits[i] = True
    r = end
    new_hypothesis = hypothesis(lm_state, bits, r, logprob, h, p)
    return new_hypothesis


def is_equal(h1, h2):
    decision = True
    if h1.lm_state != h2.lm_state:
        decision = False
    for i in range(len(h1.bitmap)):
        if h1.bitmap[i] != h2.bitmap[i]:
            decision = False
    if h1.end != h2.end:
        decision = False
    return decision


def add_to_stack(stack, h1):
    keys = stack.keys()
    for key in keys:
        h2 = stack[key]
        if is_equal(h1, h2):
            if h1.logprob > h2.logprob:
                state = h2.lm_state
                stack[state] = h1
                return stack
            else:
                return stack
    state = h1.lm_state
    stack[state] = h1
    return stack


# tm should translate unknown words as-is with probability 1
for word in set(sum(french, ())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for num, f in enumerate(french):
    sys.stderr.write("\nDecoding: %s\n" % (num,))
    all_p_phrases = get_all_phrases(f, tm)
    initial_hypothesis = hypothesis(lm.begin(), [False] * len(f), 0, 0.0, None, None)
    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis

    for i, stack in enumerate(stacks[:-1]):
        sys.stderr.write(".")
        for h in beam(stack):
            ps = ph(h, all_p_phrases)
            for next_p_phrase in ps:
                next_hypothesis = get_next_hypothesis(h, next_p_phrase, lm)
                j = get_hypothesis_length(next_hypothesis)
                stacks[j] = add_to_stack(stacks[j], next_hypothesis)

    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

    e_phrases=extract_english_phrases(winner, [], len(f))

    print " ".join(improve(e_phrases))
    #print " ".join(e_phrases)

    if opts.verbose:
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
                         (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
