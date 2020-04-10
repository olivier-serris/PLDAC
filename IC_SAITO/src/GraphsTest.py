from IC_EM_Saito2008 import IC_EM_Saito2008
from IC_EM_NotContiguous import IC_EM_NotContiguous
import Cascade as csc
import scipy
from scipy.sparse import dok_matrix
from sklearn.model_selection import cross_validate 
import numpy as np
import Metrics
import pandas as pd
import pprint as pp
from networkx import nx



def CrossVal_MAP(model,D):
    return Metrics.MAP(model.graph,D)

def handmadeGraphTest():
    handmadeGraph=dok_matrix((5,5),dtype=np.float32)
    handmadeGraph[0,1] =0.5
    handmadeGraph[0,2] =0.5
    handmadeGraph[1,3] =0.6
    handmadeGraph[2,3] =0.4
    handmadeGraph[2,4] =0.6
    
    cascades = csc.genCascades(handmadeGraph,50)
    D = [csc.CascadeToTimeRepr(c) for c in cascades]

    
    models= [IC_EM_Saito2008(),IC_EM_NotContiguous()]
    scores = dict()
    for i,model in enumerate(models):
        cross_res = cross_validate(model,D,scoring=CrossVal_MAP)
        scores[str(model)] = cross_res['test_score'].mean()
    print("score for handmadegraph")
    scores["original"] = Metrics.MAP(handmadeGraph,D)
    pp.pprint(scores)
    

def ScoresGraphModels(graphs,graphs_titles,models):
    scores = dict()
    for g,t in zip(graphs,graphs_titles) : 
        cascades = csc.genCascades(g,40)
        D = [csc.CascadeToTimeRepr(c) for c in cascades]
        scores[t] = dict()
        for model in models: 
            print("for :",t,str(model))
            
            model.set_params(csc.nodes_in_D(D))
            
            cross_res = cross_validate(model,D,scoring=CrossVal_MAP)
            scores[t][str(model)] = cross_res['test_score'].mean()
            print(cross_res)
            '''
            model.fit(D)
            print(str(model),"fitted")
            scores[t][str(model)] = Metrics.MAP(model.graph,D)
            print(str(model),f"score:{scores[t][str(model)]}")
            #'''
        scores[t]["original"] = Metrics.MAP(g,D)
    
    return pd.DataFrame(scores)


def main():
    print("start")
    #handmadeGraphTest()
    nbNodes = 50
    density = 0.07
    
    scale_free = dok_matrix(nx.to_scipy_sparse_matrix(nx.scale_free_graph(nbNodes)))
    sparseGraph = dok_matrix(scipy.sparse.random(nbNodes,nbNodes,density=density))
    graphs = [scale_free,sparseGraph]
    
    graphs_titles = [f"scale_free_{nbNodes}",f"sparseG_{nbNodes}_d : {density}"]
    models=  [IC_EM_NotContiguous(),IC_EM_Saito2008()]
    df = ScoresGraphModels(graphs,graphs_titles,models)
    print(df)
    print("end")
    

if __name__ == "__main__":
    main()