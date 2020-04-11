"""
    Cascades : 
    --> Generation of cascades.
    --> data structures for cascades
    --> Cascade filtering
"""
import numpy as np
import functools 

def genCascades(graph,nbCascades):
    '''
    Generate nbCascades of infection from graph
    Parameters
    ----------
    graph : dok_matrix
        influence graph
    nbCascades : integer
    Returns cascades, dict{nodeInfected : timeInfected}
    '''
    cascades = []
    for i in range(nbCascades):
        startNode = np.random.randint(graph.shape[0])
        cascades.append(genCascade(graph,startNode))
    return cascades


def genCascade(graph,startNode,startTime=0):
    '''
    Receive Sparse Matrix graph and starting infected node generate a cascade
    Parameters
    ----------
    graph : dok_matrix
        Graph from wich to generate cascades
    startNode : TYPE
        Id of first infected node
    startTime : TYPE, optional
        Time at wich the first infection occurs. The default is 0.
    Returns dictionnary of {infectedNode : infectedTime}
    '''
    cascade = {startNode : startTime}
    lastInfected = [startNode]
    infected_next = {}
    time = startTime+1
    while len(lastInfected)> 0:
        for infected in lastInfected:
            for (_,child),pct in graph[infected,:].items():
                if (child not in cascade) and (child not in infected_next):
                    if (np.random.rand()<pct):
                        infected_next[child] = time
        cascade.update(infected_next)
        lastInfected = list(infected_next.keys())
        infected_next = {}
        time +=1
    
    return cascade

def CascadeToTimeRepr(cascade):
    '''
    Convert a cascade C into its timed representation D
    Parameters
    ----------
    cascade : Dict
        Dictionnaray of  {infectedNode : infectedTime} 
    Returns D the timed cascade representation : 
        D[i] = all nodes infected at time i
    '''
    maxT  = max(cascade.values())+1
    Ds = [[] for i in range(maxT)]
    for (n,t) in cascade.items():
        Ds[t].append(n)
    return Ds

def nodes_in_Ds(Ds):
    '''
    Gets all unique nodes in D_s the timed representation of 
    the s cascade.
    Parameters
    ----------
    Ds : Array of Array of Nodes

    Returns np.array of nodes
    -------
    '''
    uniques = []
    for nodes in Ds : 
        for n in nodes : 
            uniques.append(n)
    return uniques


def nodes_in_D(D):
    return np.unique([n for Ds in D 
                        for nodes in Ds 
                        for n in nodes])
def sortDbySource(D):
    'Sort cascades repr by source (first infecteed node)'
    return sorted(D, key = lambda Ds:Ds[0][0])
    

def nodes_in_cascades(C):
    return np.fromiter(set().union(*[set(c.keys()) for c in C]),dtype=int)


def Episode_Where_Tu_precedes_Tv(D,u,v):
    '''
    Returns all episode id where t_u=t_v+1 
    Parameters
    ----------
    D : Array
        Containes All Ds (timed representation of cascades)
    u : Node
    v : Node
    Returns Array of episodes ids
    '''
    D_plus = []
    for i,Ds in enumerate(D): 
        for t in range(1,len(Ds)):
            if (u in Ds[t-1] and v in Ds[t]):
                D_plus.append(i)
                break
    return D_plus

def NbEpisode_Where_Tu_Not_precedes_Tv(D,u,v):
    '''
    Return Cardinal of the set of epsiodes (Ds) where node v is in episode and not(t_u=t_v+1) 
    Parameters
    ----------
    D : Array
        Containes All Ds (timed representation of cascades)
    u : Node
    v : Node
    Returns Array of episodes ids
    '''
    D_minus_len=0
    for Ds in D : 
        for t in range(1,len(Ds)):
            if (u in Ds[t-1] and v not in Ds[t]):
                D_minus_len+=1
                break
        if (u in Ds[-1]):
            D_minus_len+=1
    return D_minus_len


def Episode_Where_Tu_ancestor_Tv(D,u,v):
    '''
    Returns all episode id where t_u<t_v 
    Parameters
    ----------
    D : Array
        Containes All Ds (timed representation of cascades)
    u : Node
    v : Node
    Returns Array of episodes ids
    '''
    D_plus = []
    
    for i,Ds in enumerate(D) : 
        preceding_nodes = []
        for t in range(0,len(Ds)):
            if (v in Ds[t]):
                if (u in preceding_nodes):
                    D_plus.append(i)
                    break
            else : 
                preceding_nodes +=Ds[t]
    return D_plus

def NbEpisode_Where_Tu_Not_ancestor_Tv(D,u,v):
    '''
    Returns number of epsiodes where : node v is in episode and not(t_u<t_v) 
    Parameters
    ----------
    D : Array
        Containes All Ds (timed representation of cascades)
    u : Node
    v : Node
    Returns Array of episodes ids
    '''
    D_minus_len = 0
    for Ds in D : 
        u_in_Ds = False
        v_in_Ds = False
        for t in range(0,len(Ds)):
            if (u in Ds[t]):
                u_in_Ds = True
            if (u_in_Ds and v in Ds[t]):
                v_in_Ds = True
                break
        if (u_in_Ds and not v_in_Ds):
            D_minus_len+=1
    return D_minus_len