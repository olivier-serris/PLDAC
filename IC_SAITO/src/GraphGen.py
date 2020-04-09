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