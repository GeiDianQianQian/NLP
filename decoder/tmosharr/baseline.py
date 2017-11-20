#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="../data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="../data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="../data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase")
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None)
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  for i, stack in enumerate(stacks[:-1]):
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:5]: # prune
      for j in xrange(i+1,len(f)+1):
        if f[i:j] in tm:
          for phrase in tm[f[i:j]]:
            logprob = h.logprob + phrase.logprob
            lm_state = h.lm_state
            for word in phrase.english.split():
              (lm_state, word_logprob) = lm.score(lm_state, word)
              logprob += word_logprob
            logprob += lm.end(lm_state) if j == len(f) else 0.0
            new_hypothesis = hypothesis(logprob, lm_state, h, phrase)
            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
              stacks[j][lm_state] = new_hypothesis
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

  def extract_english_phrases(h, arr):
    if h.predecessor is None:
      return arr
    else:
      arr.append(h.phrase.english)
      return extract_english_phrases(h.predecessor, arr)

  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

  def score(arr):
    logprob = 0.0
    auxArr = arr[:]
    auxArr.insert(0, lm.begin()[0])
    for i, word in enumerate(auxArr[:-1]):
      try:
        (lm_state, word_logprob) = lm.score(tuple(word.split()[-2:]), auxArr[i+1].split()[0])
        logprob += word_logprob
      except:
        logprob += -6
    try:
      logprob += lm.end(tuple(auxArr[-1]))
    except:
      logprob += 0.0
    return logprob

  def remove(arr, s):
    for i, _ in enumerate(arr[:-2]):
      aux = arr[:]
      sc1 = score(aux[i:i+2])
      del aux[i]
      sc2 = score(aux[i-1:i+1])
      sc  = score(aux)
      if (sc2 < sc1 and sc > s + 5):
        return (aux, True)
    return (arr, False)

  def swap(arr, s):
    best = arr[:]
    aux = arr[:]
    changed = False
    for i, _ in enumerate(arr[:-1]):
      for j, _ in enumerate(arr[:-1]):
        aux[i], aux[j+1] = aux[j+1], aux[i]
        if (score(aux) > s):
          best = aux[:]
          changed = True
    return (best, changed)

  def improve(h):
    current = extract_english_phrases(winner, [])[::-1]
    while True:
      s_current = score(current)
      (current, c1) = swap(current, s_current)
      (current, c2) = remove(current, s_current)
      if not c1 or not c2:
        break
    return current

  print " ".join(improve(winner))

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
