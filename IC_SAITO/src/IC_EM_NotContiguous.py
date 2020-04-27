from IC_EM_Saito2008 import IC_EM_Saito2008
import numpy as np
import Cascade as csc


class IC_EM_NotContiguous(IC_EM_Saito2008):
    
    def P_Infected_sw(self,g,Ds,w,t):
        if t==0:
            return 1
        preceding_nodes = [] # on regroupe les noeuds des épisodes précedents
        for nodes in Ds[:t]:
            for n in nodes : 
                preceding_nodes.append(n)
            
        return 1 - np.prod ([1-g[parent,w] for parent in preceding_nodes])
    
    def __str__(self):
        return f'IC_EM_NotContiguous'
    
    def D_plus_uv_id(self,C,u,v):
        return csc.Episode_Where_Tu_ancestor_Tv(C,u,v)
    
    def D_minus_uv_len(self,C,u,v):
        return csc.NbEpisode_With_u_and_not_v(C,u,v) 
    
