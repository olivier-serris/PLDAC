import numpy as np
import Cascade as csc 

def MSE(ref,g):
    '''calcule la mean square error entre 2 graphes '''
    g = g.copy()
    g.resize(ref.shape)
    return ((ref-g).power(2)).mean()

def SSE(g1,g2):
    '''calcule la somme des erreursau carré entre 2 graphes '''
    if (g1.shape != g2.shape):
        if (g2.shape[0] > g1.shape [0]):
            g1,g2 = g2,g1
        g2 = g2.copy()
        g2.resize(g1.shape)
    return (g1-g2).power(2).sum()


def MAE(ref,g):
    '''calcule la mean absolute error entre 2 graphes '''
    g = g.copy()
    g.resize(ref.shape)
    return (np.abs(ref-g)).mean()

def Pws_gs(graph,sources,nbEpisode=500):
    ''' Calcule la probabilité qu'un noeud soit infecté connaissant 
        une source et un graph de diffusion
        On calcule par moyenne sur echantionnage d'episodes d'infections '''

    proba_infected = {n:0 for n in range(graph.shape[0])}
    for i in range(nbEpisode):
        c = csc.genCascade(graph,sources)
        for node in c : 
            proba_infected[node]+=1/nbEpisode
    return proba_infected

def AP(U_d,DsNodeSet):
    '''Average Precision pour un episode Ds '''
    ap =0
    tp = 0 # true positive
    for i,node in enumerate(U_d) : 
        if (node in DsNodeSet):
            tp+=1
            ap+= tp/(i+1) # precision = tp/(tp+fp)
    ap/= len(DsNodeSet)
    return ap

def MAP(graph,C):
    C = csc.sortCbySource(C)
    last_sources =None
    mean_ap = 0
    nb = 0
    for c in C :
        t0 = min (c.values()) # time of first infection
        new_sources = list(filter(lambda k : c[k] == t0,c))
        if new_sources != last_sources: # on actualise les predictions en fonctions de la source
            last_sources = new_sources
            pws_gs =Pws_gs(graph,last_sources)
            U_d = sorted(pws_gs,key=pws_gs.get,reverse=True) # sort par ordre decroissant
            U_d = list(filter(lambda k : k not in last_sources,U_d)) # retire les sources
        DsNodeSet =set(c.keys()).difference(last_sources)# noeuds faisant partie de l'episode d'infection sans sources
        
        if len(DsNodeSet) > 0:
            mean_ap+=AP(U_d,DsNodeSet)
            nb=nb+1
        
    return mean_ap/nb