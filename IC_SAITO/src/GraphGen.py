import numpy as np
from scipy.sparse import dok_matrix

def RandGraphFromNodes(nodes):
    '''
    Creates a random dok_matrix initializing only the edges connecting the nodes 
    passed in parameters
    ----------
    Nodes : list of Nodes

    Returns dok_matrix of edges
    -------
    graph : TYPE
        DESCRIPTION.

    '''
    m=np.max(nodes)+1
    graph = dok_matrix((m,m),dtype=np.float32)
    for n1 in nodes:
        for n2 in nodes:
            if(n1!=n2):
                graph[n1,n2]=np.random.rand()
    return graph

def randomize_edges_values(graph):
    '''
    return   copy of original dok_matrix with random value for existing egdes
    Parameters
    ----------
    graph : dok_matrix
        DESCRIPTION
    Returns
    new_g : copy of original dok_matrix with random value for existing egdes

    '''
    new_g = graph.copy().astype(np.float32)
    for key in graph.keys():
        new_g[key] = np.random.rand()
    return new_g
    