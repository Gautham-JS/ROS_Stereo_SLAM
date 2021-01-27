
res = []

def ssum(s,subset, branch, depth):
    if sum(branch)>s:
        return
    if sum(branch)==s:
        if set(branch) not in res:
            res.append(set(branch))
    if depth>1000:
        print("depth limit reached")
        return

    for i in range(len(subset)):
        subsubset = subset[i+1:]
        num = subset[i]
        depth+=1
        branchcpy = branch.copy()
        branchcpy.append(num)
        ssum(s, subsubset, branchcpy, depth)

ss = [10,30,65,80,35,21,32,35,41,100,11,5,6,8,12,150,30,63,102,12]
ss = sorted(ss)
print(ss, len(ss))
sval = 165

ssum(sval, ss, [], 0)
j = 0
for i in res:
    if sum(i)==sval:
        print(i, sum(i))
        j+=1
print(j)