'''
    Implementation of the EM algorithm to infer an influence graph from cascades of infections (Saito et al 2008)
'''
from sklearn.base import BaseEstimator
import Cascade as csc
import GraphGen as GGen
import numpy as np
import Metrics
from scipy.sparse import dok_matrix

class IC_EM_Saito2008(BaseEstimator):
    
    def __init__(self,nodes=[0],threshold = 10**(-1)):
        self.nodes = nodes
        self.threshold=threshold
        
    def fit(self,C):
        D = [csc.CascadeToTimeRepr(c) for c in C]
        self.EM_IC(C,D)
        
    def set_params(self,nodes,threshold = 10**(-1)):
        self.threshold=threshold
        self.nodes = nodes
    
    def __str__(self):
        return f'IC_EM_Saito2008'
        
################### EM algorithm ####################

    def EM_IC(self,C,D,debug = False):
        # initalisation
        self.graph = GGen.RandGraphFromNodes(self.nodes)
        D_plus_id =   {(v,u):self.D_plus_uv_id(C,v,u) for v,u in self.graph.keys()}
        D_minus_len = {(v,u):self.D_minus_uv_len(C,v,u)for v,u in self.graph.keys()}
        self.remove_edges(self.graph,D_plus_id)
        
        if debug:
            ll = -float("inf") # ll : loglikelyhood
        
        loop = True
        while loop:
            p_sw = self.Expectation(self.graph,D,C)
            next_g = self.Maximisation(self.graph,D_plus_id,D_minus_len,p_sw)
            loop = Metrics.MSE(self.graph,next_g)  > self.threshold
            self.graph = next_g
            if debug : 
                new_ll = self.llikelyhood(self.graph,D)
                if new_ll < ll : 
                    raise Exception(f"Likelyhood Error : descreasing : from {ll} to {new_ll}")
                ll = new_ll
        return self.graph

    def Expectation(self,g,D,C):
        p_sw = [{n:self.P_Infected_sw(g,Ds,n,C[i][n]) 
                 for n in C[i].keys()} 
                    for i,Ds in enumerate(D)]
        return p_sw

    def Maximisation(self,g,D_plus_id,Dminus_len,p_sw):
        ''' Calcule les nouveaux paramètres pour le graphe'''
        gprime = dok_matrix(g.shape,dtype=np.float32)
        for u,v in g.keys():
            gprime[u,v] = self.Maximisation_uv(g,D_plus_id,Dminus_len,p_sw,u,v)
        return gprime

    def Maximisation_uv(self,g,D_plus_id,Dminus_len,p_sw,u,v):
        '''Calcule les nouveaux paramètre pour l'arete u,v '''
        D_plus_id_u_v =D_plus_id.get((u,v),[])
        Dminus_len_u_v =Dminus_len.get((u,v),0)
        D_plus_u_v_len = len(D_plus_id_u_v)
        if ((D_plus_u_v_len+Dminus_len_u_v) == 0):
            #raise Exception(f"{u}-{v} Division zero")
            return 0
        return (1/(D_plus_u_v_len + Dminus_len_u_v)) * np.sum([g[u,v]/p_sw[i][v] for i in D_plus_id_u_v])

    def P_Infected_sw(self,g,Ds,w,t):
        '''
        Likelyhood of positive infection of node w given 
        D_s cascade and the influence graph g
        Parameters
        ----------
        g : dok_matrix
        Ds : Array
            Time representation of matrix.
        w : node 
        t : integer
            time of w infection in cascade D_s
        Returns Integer    
        '''
        if t==0:
            return 1
        return 1 - np.prod ([1-g[parent,w] for parent in Ds[t-1]])
    
    def D_plus_uv_id(self,C,u,v):
        return csc.Episode_Where_Tu_precedes_Tv(C,u,v)
    def D_minus_uv_len(self,C,u,v):
        return csc.NbEpisode_Where_Tu_Not_precedes_Tv(C,u,v) 
    
############# Graph Optimisation ######################

    def remove_edges(self,g,D_plus):
        ''' Retire en place tous les arêtes (u,v) ou il n'existe pas de cascade ou 
            u précède v. (=>on ne peut rien dire sur les probabilité que u infecte v)
        '''
        for (n1,n2) in list(g.keys()):
            if len(D_plus.get((n1,n2),[])) == 0:
                g[n1,n2] = 0
    
################### Likelyhood  ####################

    def P_sw(self,g,Cs,Ds,w):
            ''' Likelyhood of the infection of node w given 
                D_s cascade and the influence graph g
                Parameters
                ----------
                g : dok_matrix
                    graphe 
                Ds : Array
                    Time representation of matrix.
                w : TYPE
                    node w 
                Returns integer 
                '''
            t = Cs[w]
            if (t == 0): # si le noeud est le premier
                return 1
            if (t is None): # si le noeud n'est pas dans l'episode de diffusion
                return self.P_NotInfected_sw(g,Ds,w)
            else :  # si le noeud est dans l'épisode de diffusion
                return self.P_Infected_sw(g,Ds,w,t) 
        
    def P_NotInfected_sw(self,g,Ds,w):
        '''
        Likelyhood of negative infection of node w given 
        D_s cascade and the influence graph g
         Parameters
        ----------
        g : dok_matrix
        Ds : Array
            Time representation of matrix.
        w : node 
        t : integer
            time of w infection in cascade D_s
        Returns Integer    
        '''
        return np.prod([1-g[parent,w] for parent in csc.nodes_in_Ds(Ds)])

    def llikelyhood_Ds(self,g,Cs,Ds):
        '''Calcule la log vraisemblance d'une cascade selon le graph '''
        ll = 0
        for v,u in g.keys() :
            Psw = self.P_sw(g,Cs,Ds,u)
            Psw = np.where(Psw!=0,Psw,np.finfo(float).eps)
            ll += np.log(Psw)
        return ll
    
    def llikelyhood(self,g,D,C):
        '''log likelyhood'''
        total = 0
        for Cs,Ds in zip(C,D) : 
            total +=self.llikelyhood_Ds(g,Cs,Ds)
        return total
    