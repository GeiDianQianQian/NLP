#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
from collections import defaultdict
import itertools

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="../data/input",
                     help="File containing sentences to translate (default=../data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="../data/tm",
                     help="File containing translation model (default=../data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="../data/lm",
                     help="File containing ARPA-format language model (default=../data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int",
                     help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int",
                     help="Limit on number of translations to consider per phrase (default=20)")
optparser.add_option("-s", "--stack-size", dest="s", default=3000, type="int", help="Maximum stack size (default=3000)")
optparser.add_option("-d", "--distortion-limit", dest="d", default=8, type="int", help="Distortion limit (default=8)")
optparser.add_option("-y", "--distortion-penalty", dest="y", default=0, type="float",
                     help="Distortion penalty (default=0)")
optparser.add_option("-w", "--beam-width", dest="beam_width", default=0.5, type="float",
                     help="Beam width (default=0.5)")

opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
p_phrase = namedtuple("p_phrase", "start, end, english, logprob")
hypothesis = namedtuple("hypothesis", "lm_state, bitmap, end, logprob, predecessor, last_phrase")
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

def extract_english_without_permutation(h):
    return "" if h.predecessor is None else "%s%s " % (extract_english_without_permutation(h.predecessor), h.last_phrase.english)

def extract_english_phrases(h, phrase_list, length):
    while (True):
        if h.last_phrase is None:
            break
        phrase_list.append(h.last_phrase.english)
        if h.predecessor is not None:
            h = h.predecessor
        else:
            break
    return phrase_list


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
    perms=[]
    for i in range(len(arr[:-3])):
        for j in xrange(i + 1, len(arr[:-2])):
            for k in xrange(j + 1, len(arr[:-1])):
                ind_array=[i, j, k]
                permutations = list(itertools.permutations(ind_array, len(ind_array)))
                for n in permutations:
                    aux = arr[:]
                    aux[i], aux[j], aux[k] = aux[n[0]], aux[n[1]], aux[n[2]]
                    perms.append(aux)
    best = arr[:]
    bests = s
    changed = False
    for it in perms:
        sc=sequence_score(it)
        if sc>bests:
            bests = sc
            best = it
            changed = True
    return (best, bests, changed)


def improve(current):
    current = current[::-1]
    while True:
        s_current = sequence_score(current)
        (current, s_current, c1) = swap(current, s_current)
        if not c1:
            break
    return current


def extract_tm_logprob(h):
    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)


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


def collides(bitmap, i, j):
    for k in range(i, j + 1):
        if bitmap[k]:
            return True
    return False


def ph(h, all_p_phrases):
    s = []
    for p_phrase_list in all_p_phrases.values():
        for p_phrase in p_phrase_list:
            if abs(h.end + 1 - p_phrase.start) > opts.d or collides(h.bitmap, p_phrase.start, p_phrase.end):
                break
            s.append(p_phrase)
    return s


def get_next_hypothesis(h, p, lm, len_f):
    st=h.lm_state
    logprob = h.logprob + p.logprob
    if st == ('<s>', 'that'):
        print p.logprob
    lm_state = h.lm_state
    for english_word in p.english.split():
        (lm_state, english_word_logprob) = lm.score(lm_state, english_word)
        logprob += english_word_logprob
        if st == ('<s>', 'that'):
            print english_word_logprob
    logprob += lm.end(lm_state) if p.end >= len_f - 1 else 0.0
    logprob -= abs(h.end + 1 - p.start)*opts.y
    bits = h.bitmap[:]
    for i in range(p.start, p.end + 1): bits[i] = True
    new_hypothesis = hypothesis(lm_state, bits, p.end, logprob, h, p)
    return new_hypothesis


def add_to_stack(stack, h1):
    if (h1.lm_state, tuple(h1.bitmap), h1.end) not in stack or stack[(h1.lm_state, tuple(h1.bitmap), h1.end)].logprob < h1.logprob:
        stack[(h1.lm_state, tuple(h1.bitmap), h1.end)] = h1
    return stack


for word in set(sum(french, ())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]


sys.stderr.write("Decoding %s...\n" % (opts.input,))
for num, f in enumerate(french):
    s=opts.s
    sys.stderr.write("\nDecoding: %s\n" % (num,))
    all_p_phrases = get_all_phrases(f, tm)
    initial_bitmap=[False] * len(f)
    initial_hypothesis = hypothesis(lm.begin(), initial_bitmap, -1, 0.0, None, None)
    stacks = [{} for _ in f] + [{}]
    stacks[0][(lm.begin(), tuple(initial_bitmap), -1)] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
        sys.stderr.write('.')
        for j, h in enumerate(sorted(stack.itervalues(), key=lambda h: -h.logprob)[:s]):
            ps = ph(h, all_p_phrases)
            for phrase in ps:
                next_hypothesis=get_next_hypothesis(h, phrase, lm, len(f))
                j = get_hypothesis_length(next_hypothesis)
                stacks[j] = add_to_stack(stacks[j], next_hypothesis)
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
    #e_phrases = extract_english_phrases(winner, [], len(f))
    #sys.stderr.write('permuting: '+ str(num)+'\n')
    #print " ".join(improve(e_phrases))
    print(extract_english_without_permutation(winner))
