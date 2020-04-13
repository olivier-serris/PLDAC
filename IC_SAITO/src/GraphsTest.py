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
import GraphGen as GGen
import matplotlib.pyplot as plt

DATA_PATH = '../data/eval/'


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
    

def saveCscadesDistrib(Cascades,t):
    lens = np.array([len(c) for c in Cascades])
    hist = np.histogram(lens)[0]
    plt.title(t)
    plt.bar(np.arange(len(hist)),hist)
    plt.savefig(DATA_PATH+t+'_csc')
    plt.show()


def CrossVal_MAP(model,D):
    return Metrics.MAP(model.graph,D)

def ScoresGraphModels(graphs,graphs_titles,models,nbCascades):
    mesures = ['MSE','MAP','time']
    data = {m:{str(g_t):dict() for g_t in graphs_titles} for m in mesures}
    
    for g,t in zip(graphs,graphs_titles) : 
        cascades = csc.genCascades(g,nbCascades)
        saveCscadesDistrib(cascades,t)
        D = [csc.CascadeToTimeRepr(c) for c in cascades]
        crossVal_MSE = lambda m,d : Metrics.MSE(g, m.graph)
        
        for model in models: 
            print("\nfor :",t,str(model))
            model.set_params(csc.nodes_in_D(D))
            cross_res = cross_validate(model,D,scoring={"MAP":CrossVal_MAP,"MSE":crossVal_MSE})
            print(cross_res)
            
            data["MSE"][t][str(model)] = cross_res['test_MSE'].mean()
            data["MAP"][t][str(model)] = cross_res['test_MAP'].mean()
            data["time"][t][str(model)] = cross_res['fit_time'].mean()

        data["MAP"][t]["original"] = Metrics.MAP(g,D)

    dfs = [pd.DataFrame(data[m]) for m in mesures]
    for df,m in zip(dfs,mesures):
        df.name = m
    return dfs

def launch_test(nbCascades):
    # Generate GRAPHS 
    nx_to_dok_rand = lambda g : GGen.randomize_edges_values(
                                    dok_matrix(nx.to_scipy_sparse_matrix(g)))
    
    scale_free = nx.scale_free_graph(100)
    scale_free_dok = nx_to_dok_rand(scale_free)
    
    erdos_renyi = nx.erdos_renyi_graph(30,p=0.05)
    erdos_renyi_dok= nx_to_dok_rand(erdos_renyi)
    
    connected_cave_man = nx.connected_caveman_graph(8,5)
    connected_cave_man_dok= nx_to_dok_rand(connected_cave_man)
    
    bara = nx.barabasi_albert_graph(30,2)
    bara_dok = nx_to_dok_rand(bara)

    graphs =[scale_free,erdos_renyi,connected_cave_man,bara]
    graphs_dok = [scale_free_dok,erdos_renyi_dok,connected_cave_man_dok,bara_dok]
    graphs_titles = [f"scale_free_{scale_free_dok.shape}",
                     f"erdos_renyi{erdos_renyi_dok.shape}",
                     f"connected_cave_man({connected_cave_man_dok.shape})",
                     f"barabasi {bara_dok.shape}"]
    
    # train and test 
    models=  [IC_EM_NotContiguous(),IC_EM_Saito2008()]
    dfs = ScoresGraphModels(graphs_dok,graphs_titles,models,nbCascades)
    pd.set_option('display.max_columns', len(graphs))
    
    # Save graphs 
    for g,t in zip(graphs,graphs_titles) : 
        plt.title(t) 
        nx.draw_networkx(g,with_labels=False,node_size=40)
        plt.savefig(DATA_PATH+t+'_g')
        plt.show()
        
    # save perfs : 
    for df in dfs:
        print(f"\n{df.name}:\n{df}")
        with open(DATA_PATH+df.name,"w") as f:
            f.write(df.to_markdown())
            
def main():
    print("start")
    launch_test(100)
    print("end")
    

if __name__ == "__main__":
    main()
