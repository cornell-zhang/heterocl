import heterocl as hcl
import random
import numpy as np
import sys
from copy import *

import matplotlib.pyplot as plt
 



#memory
mc = 10.0
mf = 10.0
ec = 30.0
ef = 20.0

#correspondance tables
ord = {}
ltab = {}

#speed of comm between local and external mem
sc = 1.0
sf = 1.0

#suffix sum
ss = [0]*1000

#Available and processed nodes for branching
av = set()
pr = set()

#Track off-chip memory
remote = []

mi = 0


def gantt_plot(lst,mst,vc,vf):
    fig, gnt = plt.subplots()

    gnt.set_xlabel('microseconds since start')
    gnt.set_ylabel('CPU/FPGA')

    gnt.set_yticks([15, 25])
    # Labelling tickes of y-axis
    gnt.set_yticklabels(['CPU', 'FPGA'])
    gnt.grid(True)
    l_cpu = []
    l_fpga =[]
    for i in range(len(mst)):
        if lst[i].device == "CPU":
            l_cpu.append((mst[i], vc[i]))
        else :
            l_fpga.append((mst[i], vf[i]))
    gnt.broken_barh(l_cpu, (10, 9), facecolors ='tab:green') 
    gnt.broken_barh(l_fpga, (20, 9),facecolors ='tab:red')
    plt.savefig("gantt1.png")


#Helpers
def sort_ord(g):
    lst, output = topological_sort(g.roots)
    n = len(lst)
    ord = {}
    for i in range(n):
        ord[lst[i]] = i
    return lst, ord

def dfs_low(node, vc ,vf,ord):
    st = 0
    for c in node.children:
        if c not in ltab.keys():
            dfs_low(c, vc, vf,ord)
    for k in node.children:
        st = max(st, ltab[k.name])
    st += min(vc[ord[node]], vf[ord[node]])
    ltab[node.name] = st
    

def comp_low(roots, vc,vf,ord):
    for e in roots:
        dfs_low(e, vc, vf,ord)

#Communication size
def communication_del(g):
    l, ord = sort_ord(g)
    n = len(l)
    cm = [[0.0 for _ in range(n)] for _ in range(n)]
    def func(node, child):
        cm[ord[node]][ord[child]] = node.tensor.dtype.bits*np.prod(list(node.tensor.shape))
    g.visit(func)
    return cm


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
            flags = [in_tensor in lst for in_tensor in use.parents]
            if sum(flags) == len(use.parents):
                working_set.append(use)
    return lst, output_tensor



#EVOLUTIONARY 

def EA1p1(l,vc,vf,cm,ord,d=1):
    n = len(l)
    p = np.random.randint(2, size=n)
    def fitness(ip):
        mst = [0.0]*n
        lc = 0
        lf = 0
        for i in range(n):
            if ip[i] == 0 :
                for k in l[i].parents:
                    if ip[ord[k]] == 0:
                        mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]//d])
                    elif ip[ord[k]] == 1:
                        mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]//d] + cm[ord[k]//d][i//d])
                    else :
                        None
                mst[i] = max(mst[i],mst[lc] + vc[lc//d])
                lc = i
            elif ip[i] == 1 :
                for k in l[i].parents:
                    if ip[ord[k]] == 0:
                        mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]//d]+ cm[ord[k]//d][i//d])
                    elif ip[ord[k]] == 1:
                        mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]//d])
                mst[i] = max(mst[i],mst[lf] + vf[lf//d])
                lf = i
            else :
                None
        return max(mst[lc] + vc[lc//d], mst[lf] + vf[lf//d])

    f = fitness(p)
    cnt=0
    while cnt < 1000000:
        cnt+=1
        
        tmp = deepcopy(p)
        for i in range(n):
            if random.randint(1,n) == 1:
                tmp[i] = 1-tmp[i]
        ff = fitness(tmp)
        if ff <= f:
            #print(cnt,f,ff)
            f=ff
            p = deepcopy(tmp)
    return f, p
        


def GA(l,vc,vf,cm,ord,d=1):
    n = len(l)
    pop_size = 100
    selec_size = 100
    pb_cross = 0.8
    p = [np.random.randint(2, size=n) for _ in range(pop_size)]
    def fitness(ip):
        mst = [0.0]*n
        lc = 0
        lf = 0
        for i in range(n):
            if ip[i] == 0 :
                for k in l[i].parents:
                    if ip[ord[k]] == 0:
                        mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]//d])
                    elif ip[ord[k]] == 1:
                        mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]//d] + cm[ord[k]//d][i//d])
                    else :
                        None
                mst[i] = max(mst[i],mst[lc] + vc[lc//d])
                lc = i
            elif ip[i] == 1 :
                for k in l[i].parents:
                    if ip[ord[k]] == 0:
                        mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]//d]+ cm[ord[k]//d][i//d])
                    elif ip[ord[k]] == 1:
                        mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]//d])
                mst[i] = max(mst[i],mst[lf] + vf[lf//d])
                lf = i
            else :
                None
        return max(mst[lc] + vc[lc//d], mst[lf] + vf[lf//d])

    def onePoint(x, y):
        ch1 = [-1]*n
        ch2 = [-1]*n
        t = np.random.randint(n+1)
        for i in range (n):
            if i<t:
                ch1[i]= x[i]
                ch2[i]= y[i]
            else:
                ch2[i]= x[i]
                ch1[i]= y[i]
        return [ch1, ch2]

    f = [fitness(e) for e in p]
    #print(len(f) == pop_size)
    cnt=0
    best = -1
    while cnt < 10000:
        #print(cnt)
        cnt+=1
        
        #elitism
        ind = -1
        val = min(f)
        for i in range(0,pop_size):
            if f[i] == val :
                ind = i
                break
        elit = [e for e in p[ind]]
        #selection
        new_b = []
        while len(new_b) < selec_size:
            t1 = np.random.randint(pop_size)
            t2 = np.random.randint(pop_size-1)
            t2 += int(t2 >= t1)
            if f[t1] <= f[t2]:
                new_b.append(p[t1])
            else:
                new_b.append(p[t2])
        
        #crossover
        cro_b = []
        for i in range(0,selec_size,2):
            if np.random.choice([0,1],p=[pb_cross,1-pb_cross]) == 0:
                cro_b += onePoint(new_b[i],new_b[i+1])
            else :
                cro_b += [new_b[i],new_b[i+1]]
        
        #mutation:
        for k in range(selec_size):
            for i in range(n):
                if random.randint(1,n) == 1:
                    cro_b[k][i] = 1-cro_b[k][i]

        
        for i in range(len(cro_b)):
            for j in range(len(cro_b[0])):
                p[i][j] = cro_b[i][j]
        r = ind
        for i in range(n):
            p[r][i] = elit[i]
        f = [fitness(e) for e in p]
        #print(min(f))

    val = f[0]
    ind = 0
    for i in range(1,pop_size):
        if val > f[i] :
            val = f[i]
            ind = i

    return val, p[ind]


#BRANCH AND BOUND    

def LB1(n, mst, i, vc, vf):
    return mst[i]

def LB2(l, mst, i):
    return mst[i] +  ltab[l[i].name]

def LB3(l, mst, i, vc, vf, cm, d=1):
    if i+1 < len(l) and l[i+1] in l[i].children:
        if l[i].device=="CPU":
            return mst[i] + vc[i//d] + min(vc[(i+1)//d], vf[(i+1)//d]+cm[i//d][(i+1)//d]) + d*ss[(i+2)//d]/2
        else :
            return mst[i] + vf[i//d] + min(vf[(i+1)//d], vc[(i+1)//d]+cm[i//d][(i+1)//d]) + d*ss[(i+2)//d]/2
    elif i+1 < len(l):
        if l[i].device=="CPU":
            return mst[i] + vc[i//d] + d*ss[(i+1)//d]/2
        else :
            return mst[i] + vf[i//d] + d*ss[(i+1)//d]/2
    else:
        if l[i].device=="CPU":
            return mst[i] + vc[i//d] 
        else :
            return mst[i] + vf[i//d] 

def LB4(l, mst, i, vc, vf):
    if l[i].device=="CPU":
        return mst[i] + np.sum([vc[k] for k in range(i+1, len(l))]) 
    else :
        return mst[i] + np.sum([vf[k] for k in range(i+2, len(l))]) 
    
#Branchings available for Strong branching
def possible_branching(l,i,tab,lim=0):
    #returns list of indices on which we can branch
    res=[]
    s = set(l[:i+1])
    n = len(l)
    for k in range(i+1,min(n,i+1+lim)):
        if len(l[k].parents)==0 or tab[k] in s:
            res.append(k)
    return res

#B&B with constrained chip memory
def BnBMem(l, mst, i, vc, vf, cm, sz, mcpu, mfpga, mi,ord):

    fass, cass, tab =  [],[],[]
    lat = [0]*len(l[i].parents)
    memc = 0
    memf = 0
    remote[i] = False
    ##CPU
    mst[i]=0
    # can't start until data is available
    l[i].device = "CPU"
    for k in l[i].parents:
        # increase needed memory
        memc += sz[ord[k]]
        #handle loading time latencies
        if k.device == "CPU" and remote[ord[k]] == True:
            lat[k] = sz[ord[k]]/sc
            remote[ord[k]] = False
        elif k.device == "FPGA":
            if remote[ord[k]] == True :
                memf += sz[ord[k]]
                lat[k] = cm[ord[k]][i] + sz[ord[k]]/sf
                remote[ord[k]] = False
            else:
                lat[k] = cm[ord[k]][i]
        else:
            return None

    ofc = 0.0
    off = 0.0
    #Needs optimization : TODO : add heuristic where offload in increasing order of outdegree : sort l accordingly
    if memc + mcpu > mc:
        for k in range(i-1,-1,-1) :
            if l[k].device == "CPU" and l[k] not in l[i].parents and remote[k] == False:
                remote[ord[k]] = True
                ofc += sz[ord[k]]/sc
                memc -= sz[ord[k]]
            if memc + mcpu <= mc:
                break
    
    if mfpga + memf > mf:
        for k in range(i-1,-1,-1) :
            if l[k].device == "FPGA" and l[k] not in l[i].parents and remote[k] == False:
                remote[ord[k]] = True
                off += sz[ord[k]]/sf
                memf -= sz[ord[k]]
            if mfpga + memf <= mf:
                break
                
                
    
    # for k in l[i].parents:
    #     if k.device == "CPU":
    #         mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]])
    #     elif k.device == "FPGA":
    #         mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]] + cm[ord[k]][i])
    #     else:
    #         return None
    for k in l[i].parents:
        if k.device == "CPU":
            mst[i] = max(mst[i],mst[ord[k]] + lat[k] + ofc)
        elif k.device == "FPGA":
            mst[i] = max(mst[i],mst[ord[k]] + lat[k] + off)
        else:
            return None
        


    # can't start until the last task on CPU finishes
    for k in range(i-1,-1,-1):
        if l[k].device == "CPU":
            mst[i] = max(mst[i],mst[k] + vc[k])
            break
    
    
    v1 = 0
    if mi < LB1(len(l),mst,i,vc,vf):
        v1 = np.Infinity
    elif len(l[i].children) == 0:
        v1 = mst[i] + vc[i]
    else :
        v1, cass = BnBMem(l, mst, i+1, vc, vf,cm, sz, memc + mcpu, memf + mfpga, mi,ord)

    ##FPGA
    memc = 0
    memf = 0
    mst[i]=0
    # can't start until data is available
    l[i].device = "FPGA"
    for k in l[i].parents:
        # increase needed memory
        memf += sz[ord[k]]
        #handle loading time latencies
        if k.device == "FPGA" and remote[ord[k]] == True:
            lat[k] = sz[ord[k]]/sf
            remote[ord[k]] = False
        elif k.device == "CPU":
            if remote[ord[k]] == True :
                memc += sz[ord[k]]
                lat[k] = cm[ord[k]][i] + sz[ord[k]]/sc
                remote[ord[k]] = False
            else:
                lat[k] = cm[ord[k]][i]
        else:
            return None

    ofc = 0.0
    off = 0.0
    #Needs optimization
    if memf + mfpga > mf:
        for k in range(i-1,-1,-1) :
            if l[k].device == "FPGA" and l[k] not in l[i].parents and remote[k] == False:
                remote[ord[k]] = True
                off += sz[ord[k]]/sf
                memf -= sz[ord[k]]
            if memf + mfpga <= mf:
                break
    
    if mcpu + memc > mc:
        for k in range(i-1,-1,-1) :
            if l[k].device == "CPU" and l[k] not in l[i].parents and remote[k] == False:
                remote[ord[k]] = True
                ofc += sz[ord[k]]/sc
                memc -= sz[ord[k]]
            if mcpu + memc <= mc:
                break
                
    

    # for k in l[i].parents:
    #     if k.device == "CPU":
    #         mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]] + cm[ord[k]][i])
    #         mem += sz[ord[k]][i]
    #     elif k.device == "FPGA":
    #         mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]])
    #     else:
    #         return None

    for k in l[i].parents:
        if k.device == "CPU":
            mst[i] = max(mst[i],mst[ord[k]] + lat[k] + ofc)
        elif k.device == "FPGA":
            mst[i] = max(mst[i],mst[ord[k]] + lat[k] + off)
        else:
            return None

    # can't start until the last task on FPGA finishes
    for k in range(i-1,-1,-1):
        if l[k].device == "FPGA":
            mst[i] = max(mst[i],mst[k] + vf[k])
            break   

    v2 = 0
    if mi < LB1(len(l),mst,i,vc,vf):
        v2 = np.Infinity
    elif len(l[i].children) == 0:
        v2 = mst[i] + vf[i]
    else :
        v2,fass = BnBMem(l, mst, i+1, vc, vf,cm, sz, memc + mcpu, memf + mfpga, mi,ord)
    mi = min(mi,min(v1, v2))
    if(v1<v2):
        l[i].device = "CPU"
        tab = ["CPU"] + cass
    else:
        l[i].device = "FPGA"
        tab = ["FPGA"] + fass
    
    return min(v1, v2), tab





def BnBNoMem(l, mst, i, vc, vf, cm,mi,ord,lc,lf,d=1):
    #Not considering memory constraints : latency on communication computed without care for available space
    fass, cass, tab =  [],[],[]
    
    ##CPU
    mst[i]=0
    # can't start until data is available
    l[i].device = "CPU"
    
    for k in l[i].parents:
        if k.device == "CPU":
            mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]//d])
        elif k.device == "FPGA":
            mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]//d] + cm[ord[k]//d][i//d])
        else:
            
            return None

    # can't start until the last task on CPU finishes
    if lc != -1:
        mst[i] = max(mst[i],mst[lc] + vc[lc//d])
    # for k in range(i-1,-1,-1):
    #     if l[k].device == "CPU":
    #         mst[i] = max(mst[i],mst[k] + vc[k//d])
    #         break

    v1 = 0
    if mi < LB3(l, mst, i, vc, vf, cm, d):
        #print("cut",i)
        v1 = np.Infinity
    elif len(l[i].children) == 0:
        v1 = mst[i] + vc[i//d]
    else :
        v1, cass = BnBNoMem(l, mst, i+1, vc, vf,cm,mi,ord,i,lf,d)

    ##FPGA
    mst[i]=0
    # can't start until data is available
    l[i].device = "FPGA"
    for k in l[i].parents:
        if k.device == "CPU":
            mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]//d] + cm[ord[k]//d][i//d])
        elif k.device == "FPGA":
            mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]//d])
        else:
            return None

    # can't start until the last task on FPGA finishes
    if lf != -1:
        mst[i] = max(mst[i],mst[lf] + vf[lf//d])
    # for k in range(i-1,-1,-1):
    #     if l[k].device == "FPGA":
    #         mst[i] = max(mst[i],mst[k] + vf[k//d])
    #         break   

    v2 = 0
    if mi < LB3(l, mst, i, vc, vf, cm, d):
        #print("cut",i)
        v2 = np.Infinity
    elif len(l[i].children) == 0:
        v2 = mst[i] + vf[i//d]
        # if (min(v1,v2) - ss[0]/2)/ss[0]/2 < 0.4:
        #     print(min(v1,v2))
        
    else :
        v2,fass = BnBNoMem(l, mst, i+1, vc, vf,cm,mi,ord,lc,i,d)
    
    
    mi = min(mi,min(v1, v2))
    
    if(v1<v2):
        l[i].device = "CPU"
        tab = ["CPU"] + cass
    else:
        l[i].device = "FPGA"
        tab = ["FPGA"] + fass

    

    return min(v1, v2), tab


# def BnBStrong(l, mst, i, vc, vf, cm,mi,ord):
#     fass, cass, tab =  [],[],[]
#     av.remove(l[i])
#     pr.add(l[i])
#     #update available branchings after processing i
#     for k in l[i].children:
#         F = True
#         for p in k.parents:
#             F = F and (p in pr) and (p not in av)
#         if F:
#             av.add(k)
#     print(i)
#     ##CPU
#     id=-1
#     for e in av:
#         if  id ==-1 or vc[ord[e]] > vc[id]:
#             id = ord[e]
#     mst[i]=0
#     # can't start until data is available
#     l[i].device = "CPU"
#     for k in l[i].parents:
#         if k.device == "CPU":
#             mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]])
#         elif k.device == "FPGA":
#             mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]] + cm[ord[k]][i])
#         else:
#             return None
#     # can't start until the last task on CPU finishes (can be optimized by tracking last operation on cpu)
#     for k in range(i-1,-1,-1):
#         if l[k].device == "CPU":
#             mst[i] = max(mst[i],mst[k] + vc[k])
#             break
#     v1 = 0
#     if mi < LB2(mst,i):
#         v1 = np.Infinity
#     elif len(av) == 0:
#         v1 = mst[i] + vc[i]
#     else :
#         v1, cass = BnBStrong(l, mst, id, vc, vf,cm,mi,ord)
#     ##FPGA
#     id=-1
#     for e in av:
#         if  id ==-1 or vf[ord[e]] > vf[id]:
#             id = ord[e]
#     mst[i]=0
#     # can't start until data is available
#     l[i].device = "FPGA"
#     for k in l[i].parents:
#         if k.device == "CPU":
#             mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]] + cm[ord[k]][i])
#         elif k.device == "FPGA":
#             mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]])
#         else:
#             return None
#     # can't start until the last task on FPGA finishes
#     for k in range(i-1,-1,-1):
#         if l[k].device == "FPGA":
#             mst[i] = max(mst[i],mst[k] + vf[k])
#             break   
#     v2 = 0
#     if mi < LB2(mst,i):
#         v2 = np.Infinity
#     elif len(av) == 0:
#         v2 = mst[i] + vf[i]
#     else :
#         v2,fass = BnBStrong(l, mst, id, vc, vf,cm,mi,ord)
#     mi = min(mi,min(v1, v2))
#     if(v1<v2):
#         l[i].device = "CPU"
#         tab = ["CPU"] + cass
#     else:
#         l[i].device = "FPGA"
#         tab = ["FPGA"] + fass
#     av.add(l[i])
#     pr.remove(l[i])
#     return min(v1, v2), tab


def BnB_through(g, mst, vc, vf, cm, mi,ord,d = 2):
    lst = []
    comp_low(g.roots, vc,vf,ord)
    m = mst*d

    inputs = []
    for t in g.roots:
        for i in range(d):
            tensor = hcl.placeholder(t.tensor.op.shape,t.name+"_"+str(i),t.tensor.dtype)
            #tensor.name = t.name +"_"+str(i)
            inputs.append(tensor)
    dfg = hcl.DataflowGraph(g.name+"_thoughput", inputs)
    ed = set()
    def func(node, child) :
        if (node.name, child.name) in ed:
            return None
        for i in range(d):
            t1 = hcl.placeholder(node.tensor.op.shape,node.name+"_"+str(i),node.tensor.dtype)
            #t1.name = node.name+"_"+str(i)
            t2 = hcl.placeholder(child.tensor.op.shape,child.name+"_"+str(i),child.tensor.dtype)
            #t2.name = child.name+"_"+str(i)
            dfg.add_edge(t1,t2)
            ltab[t1.name] = ltab[node.name]
            ltab[t2.name] = ltab[child.name]
        ed.add((node.name, child.name))
    g.visit(func)
    #print([e.name for e in ltab.keys()])
    #dfg.visualize()
    #print(dfg.node_map["bn1_0"].children)
    lst, ord = sort_ord(dfg)
    for e in lst:
        if e.name not in ltab.keys():
            print(e.name)
    #print(ltab.keys())
    print([e.name for e in lst])
    
    return BnBNoMem(lst, m, 0, vc, vf, cm, mi,ord,-1,-1,d)




def partition_solve(s,sp,tc ={}, tf={}):
    g = s.DataflowGraph
    lst, ord = sort_ord(g)
    print(ord)
    n = len(lst)
    mst = [0.0]*n
    tab = [-1]*n
  
    #Costs
    vc, vf, cm, sz = [0]*n,[0]*n,[0]*n,[0]*n
    if len(tc)==0 :
        vc = [round(random.uniform(0.0,10.0),2) for _ in range(n)]
    else :
        for e in lst:
            vc[ord[e]] = tc[e.name]
    if len(tf)==0 :
        vf = [round(random.uniform(0.0,10.0),2) for _ in range(n)]
    else :
        for e in lst:
            vf[ord[e]] = tf[e.name]
    
    sz = communication_del(g)
    cm = np.true_divide(sz,sp)
    
    print("costs are")
    for i in range(n):
        print(i,vc[i],vf[i])
    print("Communication costs are :")
    for i in range(n):
        cm[i][i]=0
        print(i, cm[i])

    #Starting UB : all on one processor        
    mi = np.sum(vc)
    print("Upper bounds")
    print(mi,np.sum(vf))

    #data structure for strong branching heuristic
    for k in range(n):
        for s in lst[k].parents:
            if tab[k] == -1 or ord[tab[k]] < ord[s]:
                tab[k] = s
    
    #suffix sums:
    ss[n-1] = min(vc[n-1], vf[n-1])
    for i in range(n-2,-1,-1):
        ss[i]= ss[i+1] + min(vc[i], vf[i])
    comp_low(g.roots, vc,vf,ord)
    print("lower bound")
    print(max(ltab[lst[0].name],ss[0]/2))
    #available nodes for bramching
    for i in range(n):
        if len(lst[i].parents)==0:
            av.add(lst[i]) 
    print(av)

    d=1
    valg, deviceg = GA(lst,vc,vf,cm,ord,d)
    vale, devicee = EA1p1(lst,vc,vf,cm,ord,d)
    
    val, dev = BnB_through(g, mst, vc, vf, cm, mi,ord,d)
    if False :
        for i in range(len(lst)):
            lst[i].device = dev[i]
        print(mst)
        mst = [0.0 for _ in range(n)]
        for i in range(n):
            if lst[i].device == "CPU":
                for k in lst[i].parents:
                    if k.device == "CPU":
                        mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]])
                    elif k.device == "FPGA":
                        mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]] + cm[ord[k]][i])
                for k in range(i-1,-1,-1):
                    if lst[k].device == "CPU":
                        mst[i] = max(mst[i],mst[k] + vc[k])
                        break
            else :
                for k in lst[i].parents:
                    if k.device == "CPU":
                        mst[i] = max(mst[i],mst[ord[k]] + vc[ord[k]] + cm[ord[k]][i])
                    elif k.device == "FPGA":
                        mst[i] = max(mst[i],mst[ord[k]] + vf[ord[k]])
                    else:
                        return None
                for k in range(i-1,-1,-1):
                    if lst[k].device == "FPGA":
                        mst[i] = max(mst[i],mst[k] + vf[k])
                        break   

    #gantt_plot(lst, mst,vc,vf)

    return val, dev, valg, (valg-val)/val, vale, (vale-val)/val,mst

def partition_graph(g,tc =[], tf=[], tcm=[], tsz = []):
    lst, output = topological_sort(g.roots)
    n = len(lst)
    Pri = False
    tab = [-1]*n
    for i in range(n):
        ord[lst[i]] = i
    mst = [0.0]*n
   
    #Costs
    if len(tc) == 0:
        vc = [round(random.uniform(0.0,1000.0),2) for _ in range(n)]
    if len(tf) == 0:
        vf = [round(random.uniform(0.0,1000.0),2)for _ in range(n)]
    if len(tcm) == 0:
        cm = [[round(random.uniform(0.,10.),2) for _ in range(n)] for i in range(n)]
    if len(tsz) == 0:
        sz = [[round(random.uniform(0.,10.),2) for _ in range(n)] for i in range(n)]

    if Pri:
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
    
    #suffix sums:
    ss[n-1] = min(vc[n-1], vf[n-1])
    for i in range(n-2,-1,-1):
        ss[i]= ss[i+1] + min(vc[i], vf[i])
    
    #compute longest subpaths
    comp_low(g.roots, vc,vf,ord)

    #LB
    print("lower bound")
    print(max(ltab[lst[0].name],ss[0]/2))
    
    #available nodes for bramching
    for i in range(n):
        if len(lst[i].parents)==0:
            av.add(lst[i]) 
    print(av)

    d=1
    valg, deviceg = GA(lst,vc,vf,cm,ord,d)
    vale, devicee = EA1p1(lst,vc,vf,cm,ord,d)
    return valg, (valg-max(ltab[lst[0].name],ss[0]/2))/max(ltab[lst[0].name],ss[0]/2), vale, (vale-max(ltab[lst[0].name],ss[0]/2))/max(ltab[lst[0].name],ss[0]/2)
    #return BnBNoMem(lst, mst, 0, vc, vf, cm, mi,ord,-1,-1,1)
