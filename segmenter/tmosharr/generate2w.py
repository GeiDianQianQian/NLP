file=open("data/count_2w.txt",'r')
prior = dict()
for line in file:
    (key,value)=line.split('\t')   #skipping 6 lines
    (before,after)=key.split(" ")
    prior[before]=prior.get(before,0)+int(value)
file.close()
file=open("data/count_2_1w.txt",'w')
for key in prior.keys():
    #print(key+"\t"+str(prior[key]))
    file.write(key+"\t"+str(prior[key])+"\n")
file.close()