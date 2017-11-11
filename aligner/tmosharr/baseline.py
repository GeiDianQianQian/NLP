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
optparser.add_option("-k", "--num_epoch", dest="num_eps", default=5, type="int", help="Number of Iterations for training")

(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
f_count = defaultdict(int)
e_count = defaultdict(int)

#sys.stderr.write("Creating French and English Dictionary, storing counts for each words")
for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1
  for e_i in set(e):
    e_count[e_i] += 1

#sys.stderr.write("Determining vocabulary size of French and English Dictionary")
v_f=float(len(f_count.keys()))+1
v_e=float(len(e_count.keys()))+1
small=0.01

t1 = defaultdict(float)
t2 = defaultdict(float)

k = 0
sys.stderr.write("\nTraining IBM Model 1 (no nulls) with Expectation Maximization...")

while (k<opts.num_eps):

    sys.stderr.write("\nIteration "+str(k))
    k += 1

    sys.stderr.write("\nM step ")
    count_e  = defaultdict(float)
    count_fe = defaultdict(float)
    count_f  = defaultdict(float)
    count_ef = defaultdict(float)

    for(n, (f, e)) in enumerate(bitext):

        f.append('NULL_VALUE')
        for f_i in set(f):

            z = 0.0
            for e_j in set(e):
                if(k == 1):
                    z += 1.0 / v_f
                else:
                    z += t1[(f_i,e_j)]

            c = 0.0
            for e_j in set(e):
                if (k == 1):
                    c = (1.0/v_f)/z
                else:
                    c = t1[(f_i, e_j)]/z
                count_fe[(f_i,e_j)] = count_fe[(f_i,e_j)] + c
                count_e[(e_j)] = count_e[e_j] + c

        f=f[:-1]
        e.append('NULL_VALUE')
        for e_i in set(e):

            z = 0.0
            for f_j in set(f):
                if(k == 1):
                    z += 1.0 / v_e
                else:
                    z += t2[(e_i,f_j)]

            c = 0.0
            for f_j in set(f):
                if (k == 1):
                    c = (1.0/v_e)/z
                else:
                    c = t2[(e_i, f_j)]/z
                count_ef[(e_i,f_j)] = count_ef[(e_i,f_j)] + c
                count_f[(f_j)] = count_f[f_j] + c

    sys.stderr.write("\n"+str(len(count_fe.keys()))+" "+str(len(count_ef.keys())))
    sys.stderr.write("\nE step ")

    for (n, (f,e)) in enumerate(count_fe.keys()):

        t1[(f,e)]=(count_fe[(f,e)]+small)/(count_e[e]+small*v_f)


    for (n, (e,f)) in enumerate(count_ef.keys()):

        t2[(e,f)]=(count_ef[(e,f)]+small)/(count_f[f]+small*v_e)


sys.stderr.write("\nAligning ")

for(n, (f, e)) in enumerate(bitext):

    result = defaultdict(list)
    e.append('NULL_VALUE')
    for (i, e_i) in enumerate(e):

        bestp = 0
        bestj = None

        for (j, f_j) in enumerate(f):
            if (t2[(e_i, f_j)] > bestp):
                bestp = t2[(e_i, f_j)]
                bestj = f_j

        result[bestj].append(e_i)

    e=e[:-1]
    f.append('NULL_VALUE')
    for (i,f_i) in enumerate(f):

        bestp = 0
        bestj = 0
        best  = None

        for (j,e_j) in enumerate(e):
            if(t1[(f_i,e_j)] > bestp):
                bestp = t1[(f_i,e_j)]
                bestj = j
                best  = e_j

        if(f_i in result.keys()):
            if(best in result[f_i] and best != 'NULL_VALUE' and f_i != 'NULL_VALUE'):
                sys.stdout.write("%i-%i " % (i, bestj))

    sys.stdout.write("\n")





    #sys.stdout.write("\n")



