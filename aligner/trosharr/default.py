#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="../data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
f_count = defaultdict(int)

for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1

keys_f=f_count.keys()
v_f=float(len(keys_f))
t = defaultdict(float)
k = 0
sys.stderr.write("Training IBM Model 1 (no nulls) with Expectation Maximization...\n")
while (k<5):
    sys.stderr.write("Iteration "+str(k)+" ....................................................\n")
    k += 1
    count_e = defaultdict(float)
    count_fe= defaultdict(float)
    for(n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            z = 0.0
            for e_j in set(e):
                if(k==1):
                    z+=1.0/v_f
                else:
                    z+=t.get((f_i,e_j),0)
            for e_j in set(e):
                if (k == 1):
                    c=(1.0/v_f)/z
                else:
                    c=(t.get((f_i, e_j), 0))/z
                count_fe[(f_i,e_j)]=count_fe.get((f_i,e_j),0)+c
                count_e[(e_j)]=count_e.get(e_j,0)+c

    keys_fe=count_fe.keys()
    for (f,e) in keys_fe:
        t[(f,e)]=count_fe[(f,e)]/count_e[e]

sys.stderr.write("Aligning................................................................\n")

result=defaultdict(defaultdict)

for(n, (f, e)) in enumerate(bitext):
    for (i,f_i) in enumerate(f):
        bestp = 0
        bestj = 0
        for (j,e_j) in enumerate(e):
            if(t[(f_i,e_j)] > bestp):
                bestp = t[(f_i,e_j)]
                bestj = j
        sys.stdout.write("%i-%i " % (i, bestj))
    sys.stdout.write("\n")

