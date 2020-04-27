"""
    Cascades : 
    --> Generation of cascades.
    --> data structures for cascades
    --> Cascade filtering
"""
import numpy as np
import copy

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

def firstInfected_D(Ds):
    for nodes in Ds: 
        if (nodes):
            return nodes[0]
    raise Exception("Empty Cascade")
        
def sortDbySource(D):
    'Sort cascades repr by source (first infected node)'
    return sorted(D, key = lambda Ds:firstInfected_D(Ds))

def sortCbySource(C):
    'Sort cascades repr by source (first infected node)'
    return sorted(C, key = lambda c: min(c, key=c.get))

def remove_xpct_infections(C, pct):
    assert 0 <= pct <=1
    ''' Returns a copy of all cascades from database from which @pct of the observable infections has been removed.'''
    infection_ids = np.array([[i,n_i] for i,c_i in enumerate(C)
                                        for n_i in c_i.keys()])
    np.random.shuffle(infection_ids)
    to_remove = infection_ids[:int(pct*len(infection_ids))]
    new_C = copy.deepcopy(C)
    for (c_i,n_i) in to_remove:
        new_C[c_i].pop(n_i)
    new_C = list(filter(bool,new_C))
    return new_C

def remove_Xpct_users(nodes,C,pct):
    ''' Returns a copy of all cascades from database from each cascades, @pct of the user are randomly selected and removed'''
    assert 0 <= pct <=1
    new_C = copy.deepcopy(C)
    for c in new_C:
         nbNodeToRemove = int(round(len(nodes)*pct))
         to_remove = np.random.choice(nodes, size=nbNodeToRemove)
         for node in to_remove : 
             if node in c:
                 c.pop(node)
    new_C = list(filter(bool,new_C))
    return new_C
    

def nodes_in_cascades(C):
    return np.fromiter(set().union(*[set(c.keys()) for c in C]),dtype=int)

def Episode_Where_Tu_precedes_Tv(C,u,v):
    '''
    Returns all episode id where t_u=t_v+1 
    Parameters
    ----------
    C : Array
        Containes All cascades
    u : Node
    v : Node
    Returns Array of episodes ids
    '''
    return [i for i,cascade in enumerate(C) 
                if u in cascade and v in cascade and cascade[u] == cascade[v]-1]

def NbEpisode_Where_Tu_Not_precedes_Tv(C,u,v):
    '''
    Return Cardinal of the set of cascades  where node u is in episode and not(t_u=t_v+1) 
    Parameters
    ----------
    C : Array
        Containes All cascades
    u : Node
    v : Node
    Returns Array of episodes ids
    '''
    return sum (u in cascade and ( v not in cascade or (cascade[u] != cascade[v]-1))
                    for cascade in C)

def Episode_Where_Tu_ancestor_Tv(C,u,v):
    '''
    Returns all casacdes id where t_u<t_v 
    Parameters
    ----------
    C : Array
        Containes All cascades
    u : Node
    v : Node
    Returns Array of episodes ids
    '''
    return [i for i,cascade in enumerate(C) 
                if u in cascade and v in cascade and cascade[u] < cascade[v]]

def NbEpisode_With_u_and_not_v(C,u,v):
    '''
    Returns number of epsiodes where : node u is in episode and not V 
    Parameters
    ----------
    C : Array
        Containes All cascades
    u : Node
    v : Node
    Returns Array of episodes ids
    '''
    return sum(u in cascade and v not in cascade 
               for cascade in C)
