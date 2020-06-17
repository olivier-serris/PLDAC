import numpy as np

    
def exp(rate):
    x=np.random.rand(*rate.shape)
    return -np.log(1.0-x)/np.where(rate==0,1e-200,rate)

def CTIC_Cascade(graph,infections,max_obs,maxT = 10):
    names,gk,gr=graph
    nbNodes=len(names)
    infectious=np.array([maxT]*nbNodes,dtype=float) # 1 case par tps et par noeuds  
    ids,t=map(np.array,zip(*infections)) # ids : id des noeuds infecté, t : temps des noeuds infectés
    ids=ids[t<=max_obs] # on garde que les les ids don les temps < max_obs
    t=t[t<=max_obs] # on garde que les tps <max_obs
    infectious[ids]=t # infectious[node_id] = tps_infecté
    times=np.copy(infectious) 
    inf=np.copy(infectious)
    while True:
        qui=np.argmin(inf)  
        quand=inf[qui]  
        inf[qui]=maxT
        if quand==maxT: # condition d'arrêt : plus de noeud à parcourir
            break
        times[qui]=quand
        vers=(times>quand) # boolean mask de tous les temps des noeuds suivants le noeuds infecté
        versq=np.arange(len(times))[vers] # ids des noeuds suivants infectés
        k=np.array([gk.get((qui, v),0) for v in versq]) 
        r=np.array([gr.get((qui, v),0) for v in versq]) 
        x=np.random.rand(len(versq))
        if quand<max_obs:
            kn=k*np.exp(-r*(max_obs-quand))
            k=kn/(kn+1-k)
        ok=x<k # tous les noeuds effectivement infectés
        t=exp(r)
        if quand<max_obs:
            quand=max_obs  
        t=(t+quand)*ok # les temps samplés pour les noeuds qu'on a reussi à infecter
        wt = ok *(t<inf[versq])# masque boolean des noeuds infectés a un nouveau temps plus faible 
        inf[versq[wt]]=t[wt] # mise à jour des noeuds infectés
    return(times)


