data = [-1,9,0,8,-5,6,-24]
prefix = [-1]

for i in range(1,len(data)):
    prefix.append(prefix[len(prefix)-1]+data[i])

def tong(u,v):
    return prefix[v]-prefix[u]+data[u]

for i in range(len(data)):
    for j in range(i,len(data)):
        print(tong(i,j),i,j)