# this script prepare the data on wseg set in the same format of the count_1w.txt
import sys, codecs, optparse, os

optparser = optparse.OptionParser()
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('.', 'wseg_simplified_cn'), help="input file to segment")
(opts, _) = optparser.parse_args()

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

with open(opts.input) as f:
    dict = {}
    for line in f:
        utf8arr = unicode(line.strip(), 'utf-8').split() # split on blank arrays
        for key in utf8arr:
            value = 1 if key not in dict else dict[key] + 1 
            dict[key] = value

    for key, value in dict.items():
        print (key + "" + "\t" + str(value) + "\n")

sys.stdout = old
