#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
import itertools

def get_list_of_phrases(phrase):
    english=phrase.english
    logprob=phrase.logprob
    phrases=[]
    words=english.split()
    if len(words)>4:
        phrases.append(phrase)
        return phrases
    else:
        permutations=list(itertools.permutations(words, len(words)))
        for permutation in permutations:
            new_english=" ".join(permutation)
            new_phrase=models.phrase(new_english,logprob)
            phrases.append(new_phrase)
        return phrases

phrase_1=models.phrase('i will be there', 0.03)
phrases=get_list_of_phrases(phrase_1)
for phrase in phrases:
    print(phrase.english)
    print(phrase.logprob)