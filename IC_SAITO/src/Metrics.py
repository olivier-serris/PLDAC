import numpy as np
import Cascade as csc 

def MSE(ref,g):
    '''calcule la mean square error entre 2 graphes '''
    g.resize(ref.shape)
    return ((ref-g).power(2)).sum()/len(ref)

def MAE(ref,g):
    '''calcule la mean absolute error entre 2 graphes '''
    g.resize(ref.shape)
    return (np.abs(ref-g)).sum()/len(ref)

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


def AP(U_d,DsNodeSet,graph):
    '''Average Precision pour un episode Ds '''
    ap =0
    for i,node in enumerate(U_d) : 
        if (node in DsNodeSet):
            tp = len(DsNodeSet.intersection(U_d[:(i+1)])) # true positive
            ap+= tp/(i+1) # precision = tp/(tp+fp)
    ap/= len(DsNodeSet)
    return ap

def MAP(graph,D):
    D = csc.sortDbySource(D)
    last_source =None
    mean_ap = 0
    for Ds in D: 
        if csc.firstInfected(Ds) != last_source: # on actualise les predictions en fonctions de le la source
            last_source = csc.firstInfected(Ds)
            pws_gs =Pws_gs(graph,csc.firstInfected(Ds))
            U_d = sorted(pws_gs,key=pws_gs.get,reverse=True) # sort par ordre decroissant
        DsNodeSet = set(csc.nodes_in_Ds(Ds)) # noeuds faisant partie de l'episode d'infection Ds
        mean_ap+=AP(U_d,DsNodeSet,graph)
    
    return mean_ap/len(D)