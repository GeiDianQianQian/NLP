"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair.

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""

import perc
import sys, optparse, os
from collections import defaultdict

def perc_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    # insert your code here
    cumulative_feat_vec=defaultdict(float)
    epoch = 0
    count = 0
    while (epoch < numepochs):
        print(epoch)
        mistakes = 0
        correct = 0
        #print(len(train_data))
        sen=0
        for sentence_data in train_data:
            words = []
            postags = []
            truetags = []
            label_list = sentence_data[0]
            feat_list = sentence_data[1]
            for label in label_list:
                (word, postag, chunktag) = label.split(" ")
                words.append(word)
                postags.append(postag)
                truetags.append(chunktag)
            tagset = perc.read_tagset(opts.tagsetfile)
            default_tag = tagset[0]
            argmaxtags = perc.perc_test(feat_vec, label_list, feat_list, tagset, default_tag)
            feat_index = 0
            i = 0

            for word in words:
                (feat_index, feats_for_this_word) = perc.feats_for_word(feat_index, feat_list)
                # print(len(feats_for_this_word))
                argmax = argmaxtags[i]
                tru = truetags[i]
                if (argmax == tru):
                    i += 1
                    continue
                for f in feats_for_this_word:
                    wrongkey = f, argmax
                    rightkey = f, tru
                    feat_vec[wrongkey] = feat_vec.get(wrongkey, 0) - 1
                    feat_vec[rightkey] = feat_vec.get(rightkey, 0) + 1
                i += 1
            i = 0

            for word in words:
                argmax = argmaxtags[i]
                tru = truetags[i]
                if (argmax == tru):
                    i += 1
                    correct += 1
                    continue
                else:
                    mistakes += 1
                argmaxprev = "B:"
                truprev = "B:"
                if (i == 0):
                    argmaxprev += "B_-1"
                    truprev += "B_-1"
                else:
                    argmaxprev += argmaxtags[i - 1]
                    truprev += truetags[i - 1]
                wrongkey = argmaxprev, argmax
                rightkey = truprev, tru
                feat_vec[wrongkey] = feat_vec.get(wrongkey, 0) - 1
                feat_vec[rightkey] = feat_vec.get(rightkey, 0) + 1
                i += 1

            keys=feat_vec.keys()
            for key in keys:
                cumulative_feat_vec[key]=cumulative_feat_vec.get(key,0)+feat_vec[key]
            count+=1

            if(sen%1000==0):
                print(str(sen)+"/"+str(len(train_data)))
            sen+=1

        #print(mistakes)
        #print(correct)
        epoch += 1

    keys = cumulative_feat_vec.keys()
    for key in keys:
        cumulative_feat_vec[key] = float(cumulative_feat_vec[key])/float(count)

    # please limit the number of iterations of training to n iterations

    return cumulative_feat_vec

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print >>sys.stderr, "reading data ..."
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print >>sys.stderr, "done."
    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)

