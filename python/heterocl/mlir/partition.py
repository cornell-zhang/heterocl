import heterocl as hcl
import random
import numpy as np
import sys


mi = 0

def topological_sort(roots):
    lst = []
    output_tensor = []
    working_set = roots.copy()
    while len(working_set) != 0:
        node = working_set.pop(0)
        lst.append(node)
        if len(node.children) == 0:  
            output_tensor.append(node)
        for use in node.children:
            flags = [
                in_tensor in lst for in_tensor in use.parents]
            if sum(flags) == len(use.parents):
                working_set.append(use)
    return lst, output_tensor

def LB1(n, mst, i, vc, vf):
    return mst[i]

def LB2(n, mst, i, vc, vf):
    return mst[i] +  np.sum([min(vc[k], vf[k]) for k in range(i, n)]) ##Can be optimized by precomputing suffix sums

def LB3(l, mst, i, vc, vf, cm):
    if i+1 < len(l) and l[i+1] in l[i].children:
        if l[i].device=="CPU":
            return mst[i] + vc[i] + min(vc[i+1], vf[i+1]+cm[i][i+1]) + np.sum([min(vc[k], vf[k]) for k in range(i+2, len(l))]) 
        else :
            return mst[i] + vf[i] + min(vf[i+1], vc[i+1]+cm[i][i+1]) + np.sum([min(vc[k], vf[k]) for k in range(i+2, len(l))]) 
    elif i+1 < len(l):
        if l[i].device=="CPU":
            return mst[i] + vc[i] + np.sum([min(vc[k], vf[k]) for k in range(i+1, len(l))]) 
        else :
            return mst[i] + vf[i] + np.sum([min(vc[k], vf[k]) for k in range(i+1, len(l))])
    else:
        if l[i].device=="CPU":
            return mst[i] + vc[i] 
        else :
            return mst[i] + vf[i] 

def LB4(l, mst, i, vc, vf):
    if l[i].device=="CPU":
        return mst[i] + np.sum([vc[k] for k in range(i+1, len(l))]) 
    else :
        return mst[i] + np.sum([vf[k] for k in range(i+2, len(l))]) 
    
#Branchings available for strong branching heuristic
def possible_branching(l,i,tab,lim=0):
    #returns list of indices on which we can branch
    res=[]
    s = set(l[:i+1])
    n = len(l)
    for k in range(i+1,min(n,i+1+lim)):
        if len(l[k].parents)==0 or tab[k] in s:
            res.append(k)
    return res


def BnB(l, mst, i, vc, vf, cm,mi,ord):

    fass, cass, tab =  [],[],[]
    ##CPU
    mst[i]=0
    l[i].device = "CPU"
    for k in l[i].parents:
        if k.device == "CPU":
            mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]])
        elif k.device == "FPGA":
            mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]] + cm[ord[k]][i])
        else:
            return None
    v1 = 0
    if mi < LB1(len(l),mst,i,vc,vf):
        v1 = np.Infinity
    elif len(l[i].children) == 0:
        v1 = mst[i] + vc[i]
    else :
        v1, cass = BnB(l, mst, i+1, vc, vf,cm,mi,ord)

    ##FPGA
    mst[i]=0
    l[i].device = "FPGA"
    for k in l[i].parents:
        if k.device == "CPU":
            mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]] + cm[ord[k]][i])
        elif k.device == "FPGA":
            mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]])
        else:
            return None
            
    v2 = 0
    if mi < LB1(len(l),mst,i,vc,vf):
        v2 = np.Infinity
    elif len(l[i].children) == 0:
        v2 = mst[i] + vf[i]
    else :
        v2,fass = BnB(l, mst, i+1, vc, vf,cm,mi,ord)
    mi = min(mi,min(v1, v2))
    if(v1<v2):
        l[i].device = "CPU"
        tab = ["CPU"] + cass
    else:
        l[i].device = "FPGA"
        tab = ["FPGA"] + fass
    
    return min(v1, v2), tab



def partition_solve(s):

    g = s.DataflowGraph
    lst, output = topological_sort(g.roots)
    n = len(lst)
    ord = {}
    tab = [-1]*n
    for i in range(n):
        ord[lst[i]] = i
    mst = [0.0]*n
   

    #Costs
    vc = [round(random.uniform(0.0,10.0),2) for _ in range(n)]
    vf = [round(random.uniform(0.0,10.0),2)for _ in range(n)]
    cm = [[round(random.uniform(0.,2.),2) for _ in range(n)] for i in range(n)]

    print("costs are")
    for i in range(n):
        print(i,vc[i],vf[i])
    print("Communication costs are :")
    for i in range(n):
        cm[i][i]=0
        print(i, cm[i])

    #Solution which will be as upper bound        
    mi = np.sum(vc)
    print("Upper bounds")
    print(mi,np.sum(vf))

    #data structure for strong branching heuristic
    for k in range(n):
        for s in lst[k].parents:
            if tab[k] == -1 or ord[tab[k]] < ord[s]:
                tab[k] = s



    return BnB(lst, mst, 0, vc, vf, cm,mi,ord)