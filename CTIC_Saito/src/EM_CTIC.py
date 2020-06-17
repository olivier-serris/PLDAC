import numpy as np
import utils as utils
import pprint as pp
from scipy.sparse import dok_matrix
'''
Formula from paper : 
Learning Continuous-Time Information Diﬀusion Model for Social Behavioral Data Analysis

Recap : 
--> A_m,u,v   Probability density that node u activate node v at time t_m,v 
--> B_u,v =   Probability that node vis not activated from node u within [tm,u,tm,v] 
--> Alpha_muv
--> Beta_muv
--> k_uv
--> r_uv
'''
# TODO transformer M_Minus en np_array 2D 

class EM_CTIC:
    def __init__(self,nodes,maxT):
        self.set_params(nodes,maxT)

    def set_params(self,nodes,maxT):
        self.nodes = nodes
        self.maxT = maxT
        
    def EM(self, mtimes):
        np.random.seed(2)

        k_uv = utils.RandGraphFromNodes(self.nodes)
        r_uv = utils.RandGraphFromNodes(self.nodes,val_range = (1,10)) # attention range totalement arbitarire
        graph = (self.nodes,k_uv,r_uv)
        
        m_uv_plus = self.M_uvPlus(graph,mtimes)
        m_uv_minus = self.M_uvMinusLen(graph,mtimes)

        self.remove_edges(k_uv,r_uv,m_uv_plus)

        preds = utils.get_preds(self.nodes,r_uv.keys())

        for _ in range(100):
            pp.pprint(f"step {_}")
            alpha_uv,beta_uv = self.Expectation(graph,preds,mtimes,m_uv_plus)
            k_uv,r_uv = self.Maximisation(graph,mtimes,alpha_uv,beta_uv,m_uv_plus,m_uv_minus)
            graph = (self.nodes,k_uv,r_uv)
            if len([1 for u,v in k_uv.keys() if k_uv[(u,v)]>1 ]):
                raise Exception()
        return graph
    
    def remove_edges(self,k_uv,r_uv,m_uv_plus):
        ''' Retire en place tous les arêtes (u,v) ou il n'existe pas de cascade ou 
            u précède v. (=>on ne peut rien dire sur les probabilité que u infecte v)
        '''
        for (n1,n2) in list(m_uv_plus.keys()):
            if len(m_uv_plus.get((n1,n2),[])) == 0:
                k_uv[n1,n2] = 0
                r_uv[n1,n2] = 0
                del m_uv_plus[(n1,n2)]

#########################################
######  Expectation2  ####################

    def Expectation(self,graph,preds,times_m,m_plus_uv):
        '''
            Compute alpha_muv for each Mplus_uv and  beta_uv for each Mplus_uv .
            return dictionnary of alpha[(u,v)] -> array of alpha_muv values for each Mplus_uv
            return dictionnary of alpha[(u,v)] -> array of alpha_muv values for each Mplus_uv
        '''
        _,_,r_uv = graph 
        alpha_uv,beta_uv = dict(),dict()
         
        alpha_den_mv = dict()

        for (u,v) in r_uv.keys():
            alpha_uv[(u,v)] = []
            beta_uv[(u,v)] = []
            for m in m_plus_uv[(u,v)]:
                if ((m,v) not in alpha_den_mv):
                    alpha_den_mv[(m,v)] = self.Alpha_denominator(v,times_m[m],graph,preds)
                alpha,beta = self.AlphaBeta(u,v,graph,times_m[m],alpha_den_mv[(m,v)])
                alpha_uv[(u,v)].append(alpha)
                beta_uv[(u,v)].append(beta)
            alpha_uv[(u,v)] = np.array(alpha_uv[(u,v)])
            beta_uv[(u,v)] = np.array(beta_uv[(u,v)])

        return alpha_uv,beta_uv

    def AlphaBeta(self,u,v,graph,times,alpha_den):
        _,k_uv,r_uv = graph
        k,r = k_uv[(u,v)],r_uv[(u,v)]
        
        dt = times[v] - times[u]

        A = k * r * np.exp(-r * dt)
        B = k * np.exp(-r * dt) + 1 - k

        beta = k *np.exp(-r *dt)/B
        alpha_numerator = A/B
        alpha = alpha_numerator/alpha_den
        # print(f"alpha {(u,v)}= ",alpha)
        # print(f"with ({A}/{B}) /{alpha_den}")
        return alpha,beta

    def Alpha_denominator(self,v,times,graph,preds):
        ''' compute the denominator in the alpha formula for node v'''
        _,k_uv,r_uv = graph
        t_v = times[v]
        preds = preds.get(v,[])
        if len(preds) > 0:
            preds_mask = np.isin(np.arange(len(times)),preds)
            infected_pred_mask = (times < t_v ) * ( times >= 0) * ( times < self.maxT) *preds_mask# mask des predecesseurs qui peuvent infecter v
            v_preds_t = times[infected_pred_mask]
            k = k_uv.getcol(v).toarray().flatten()[infected_pred_mask]
            r = r_uv.getcol(v).toarray().flatten()[infected_pred_mask]
            dt = t_v - v_preds_t

            A_xv = k * r * np.exp(-r*dt)
            B_xv = k * np.exp(-r* dt)+1-k #TODO on peut re-découper les formules pour opti 
            alpha_den = sum (A_xv/B_xv)
        else : 
            raise Exception("not supposed to be computed")
        return alpha_den    
    
#########################################
######  Maximisation  ####################

    def Maximisation(self,graph,mtimes,alpha_uv,beta_uv,m_uv_plus,m_uv_minus):
        ''' return new graph params : k_uv and r_uv'''
        _,_,r_uv = graph
        new_k_uv,new_r_uv = dok_matrix(r_uv.shape),dok_matrix(r_uv.shape)
        for u,v in r_uv.keys():
            k,r = self.K_uv_And_R_uv(u,v,mtimes,alpha_uv,beta_uv,m_uv_plus,m_uv_minus)
            new_k_uv[(u,v)] = k
            new_r_uv[(u,v)] = r
        return new_k_uv,new_r_uv
        
    def K_uv_And_R_uv(self,u,v,mtimes,alpha_uv,beta_uv,m_uv_plus,m_uv_minus):
        alpha_mplus = alpha_uv[(u,v)]
        beta_mplus = beta_uv[(u,v)]
        times_mplus = mtimes[m_uv_plus[(u,v)]]

        dt =  times_mplus[:,v] - times_mplus[:,u] 

       # compute k_uv
        nominator = np.sum(alpha_mplus + (1-alpha_mplus) * beta_mplus)
        denominator = len(m_uv_plus[(u,v)])+m_uv_minus[u,v]
        k_uv = nominator/denominator        

        # compute r_uv
        nominator = np.sum(alpha_mplus)
        denominator = np.sum((alpha_mplus + (1-alpha_mplus)*beta_mplus)*dt)
       
        r_uv = nominator / denominator

        return k_uv,r_uv

#########################################
######  M_Plus / M_minus  ###############

    def M_uvPlus(self,graph,times):
        ''' 
            return M_{u,v}^{Plus}  : 
            ex : M_uvPlus[(u,v)] = [1,2, .. ]
            return type : dict
        '''
        _,_,r_uv= graph
        return  {(u,v) : self.M_Plus(u,v,times) for (u,v) in r_uv.keys() } 
        
    
    def M_Plus(self,u,v,times):
        '''
            return  M_{u,v}^{Plus} : for nodes (u,v), the id of episodes where tm,u < tm,v
            u : node
            v : node 
            times[m][n] : time of node n in cascade m
            return array of diffusion episode id 
        '''
        times_v = times[:,v]
        times_u = times[:,u]
        tu_before_tv_mask = (times_u < times_v) * (( times_u >= 0) * ( times_v < self.maxT)) *( times_v >= 0)
        return np.arange(len(times))[tu_before_tv_mask]
    
    def M_uvMinusLen(self,graph,times):
        ''' 
            return M_{u,v}^{Minus}  : 
            ex : M_uvPlus[(u,v)] = [1,2, .. ]
            return type : dict
        '''
        _,_,r_uv= graph
        return  {(u,v) : self.M_MinusLen(u,v,times) for (u,v) in r_uv.keys() }

    def M_MinusLen(self,u,v,mtimes):
        '''return  M_{u,v}^{Minus} : for nodes (u,v), the number of episodes where u in Episode_m and v not in Epsiode_m 
            u : node
            v : node 
            times[m][n] : time of node n in cascade m
            return  nb of diffusion episode id (int)
        '''
        times_v = mtimes[:,v]
        times_u = mtimes[:,u]
        u_not_v_mask = ((times_u >= 0) * ( times_u < self.maxT)) * (( times_v < 0)+( times_v >= self.maxT))
        return np.sum(u_not_v_mask)