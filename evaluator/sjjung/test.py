

s2='Look for alternatives in cities'.split(' ')
s1 ='Looking for alternatives in the cities'.split(' ')
mylist=[]
while(lcs(s2,s1) != []):
    x = lcs(s2, s1)
    mylist.append(x)
    s1 = [i for i in s1 if i not in x]
    s2 = [i for i in s2 if i not in x]

print mylist