import numpy as np
from scipy.sparse import dok_matrix

def RandGraphFromNodes(nodes,val_range = (0,1)):
    '''
    Creates a random dok_matrix initializing only the edges connecting the nodes 
    passed in parameters
    ----------
    Nodes : list of Nodes
    val_range : [min,max]
    Returns dok_matrix of edges
    -------
    graph : TYPE
        DESCRIPTION.

    '''
    min_val,max_val = val_range
    m=np.max(nodes)+1
    graph = dok_matrix((m,m),dtype=np.float64)
    for n1 in nodes:
        for n2 in nodes:
            if(n1!=n2):
                graph[n1,n2]=round(np.random.rand()*(max_val-min_val) + min_val,2)
    return graph

def get_preds(nodes,edges):
    preds = dict()
    for (u,v) in edges : 
        preds[v] = preds.get(v,[]) +[u]
    for v in preds :
        preds[v] = np.array(preds[v])
    return preds

def print_graph(g):
    _,k_uv,r_uv = g 
    print("graph : ")
    for u,v in k_uv.keys():
        print(f"{u,v} k={k_uv[(u,v)]},r={r_uv[(u,v)]}" )