import numpy as np
import Cascade as csc 

def MSE(g1,g2):
    '''calcule la mean square error entre 2 graphes '''
    assert g1.shape == g2.shape
    return np.sum((g1-g2).power(2))


def Pws_gs(graph,source,nbEpisode=100):
    ''' Calcule la probabilité qu'un noeud soit infecté connaissant 
        une source et un graph de diffusion
        On calcule par moyenne sur echantionnage d'episodes d'infections '''

    proba_infected = {n:0 for n in range(graph.shape[0])}
    for i in range(nbEpisode):
        c = csc.genCascade(graph,source)
        for node in c : 
            proba_infected[node]+=1/nbEpisode
    return proba_infected


def AP(Ds,graph):
    '''Average Precision pour un episode Ds '''
    pws_gs =Pws_gs(graph,Ds[0][0])
    U_d = sorted(pws_gs,key=pws_gs.get,reverse=True) # sort par ordre decroissant
    DsNodeSet = set(csc.nodes_in_Ds(Ds)) # noeuds faisant partie de l'episode d'infection Ds
    ap =0
    for i,node in enumerate(U_d) : 
        if (node in DsNodeSet):
            tp = len(DsNodeSet.intersection(U_d[:(i+1)])) # true positive
            ap+= tp/(i+1) # precision = tp/(tp+fp)
    ap/= len(DsNodeSet)
    return ap

def MAP(graph,D):
    return np.sum([AP(Ds,graph)for Ds in D])/len(D)