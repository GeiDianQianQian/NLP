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

bitext = [[sentence.lower().strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

sys.stderr.write("\nGetting size vocabulary f")
count_f = defaultdict(float)
count_e = defaultdict(float)
for (n, (f, e)) in enumerate(bitext):
    for f_i in set(f):
        count_f[f_i] += 1
    for e_j in set(e):
        count_e[e_j] += 1
    if n % 500 == 0:
        sys.stderr.write(".")

def similarity(f_w, e_w):
    cnt = 0.0
    for f_ch in f_w:
        for e_ch in e_w:
            if (f_ch == e_ch):
                cnt += 1
                break
    return cnt / len(f_w)

sys.stderr.write("\nApplying model")
v_f = float(len(count_f.keys()))
v_e = float(len(count_e.keys()))
small = 0.01
t1 = defaultdict(float)
t2 = defaultdict(float)
k = 0

while (k < opts.num_eps):
    k += 1
    sys.stderr.write("\nIteration " + str(k))
    count_e  = defaultdict(float)
    count_fe = defaultdict(float)
    count_f  = defaultdict(float)
    count_ef = defaultdict(float)

    for(n, (f, e)) in enumerate(bitext):
        e.append("n_wd")
        f.append("n_wd")
        for (i, f_i) in enumerate(f):
            z = 0.0
            for (j, e_j) in enumerate(e):
                if k == 1:
                    if(abs(i - j) <= 3): 
                        sim_rate = similarity(f_i, e_j) * 2
                        v_f += sim_rate
                        z += (1.0 + sim_rate) / v_f
                    else:
                        z += 1.0 / v_f
                else:
                    z += t1[(f_i, e_j)]

            c = 0.0
            for e_j in set(e):
                c = t1[(f_i, e_j)] / z if k > 1 else 1.0
                count_fe[(f_i, e_j)] = count_fe[(f_i, e_j)] + c
                count_e[(e_j)] = count_e[e_j] + c
        
        for e_i in set(e):
            z = 0.0
            for f_j in set(f):
                z += t2[(e_i, f_j)] if k > 1 else 1.0 / v_e

            c = 0.0
            for f_j in set(f):
                c = t2[(e_i, f_j)] / z if k > 1 else 1.0
                count_ef[(e_i, f_j)] = count_ef[(e_i, f_j)] + c
                count_f[(f_j)] = count_f[f_j] + c

        if n % 500 == 0:
            sys.stderr.write(".")

    sys.stderr.write("\nGetting translation rate")

    for (n, (f, e)) in enumerate(count_fe.keys()):
        t1[(f, e)] = (count_fe[(f, e)] + small) / (count_e[e] + small * v_f)
    for (n, (e, f)) in enumerate(count_ef.keys()):
        t2[(e, f)] = (count_ef[(e, f)] + small) / (count_f[f] + small * v_e)

sys.stderr.write("\nAligning ")

for(n, (f, e)) in enumerate(bitext):
    e.append("n_wd")
    f.append("n_wd")
    result = defaultdict(list)

    for (i, e_i) in enumerate(e):
        bestp = 0
        bestj = 0
        for (j, f_j) in enumerate(f):
            curp = t2[(e_i, f_j)]
            if(curp > bestp):
                bestp = curp
                bestj = j
                word = f_j
            elif(curp == bestp):
                if (abs(i - j) < abs(i - bestj)):
                    bestp = curp
                    bestj = j
                    word = f_j

        # align e_aj to f_j
        result[i] = bestj

    for (i, f_i) in enumerate(f):
        bestp = 0
        bestj = 0
        word  = None
        for (j, e_j) in enumerate(e):
            curp = t1[(f_i, e_j)]
            if(curp > bestp):
                bestp = curp
                bestj = j
                word = e_j
            elif(curp == bestp):
                if (abs(i - j) < abs(i - bestj)):
                    bestp = curp
                    bestj = j
                    word = e_j

        # if f_j matches to e_aj, then we are sure about it
        if result[bestj] == i and word != "n_wd":
            sys.stdout.write("%i-%i " % (i, bestj))
        # otherwise we are not so sure about it
        elif word != "n_wd":
            if similarity(f_i, word) >= 0.75:
                sys.stdout.write("%i-%i " % (i, bestj))

    sys.stdout.write("\n")
    if n % 500 == 0:
        sys.stderr.write(".")
