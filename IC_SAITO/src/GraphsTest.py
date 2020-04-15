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
import time

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
    plt.hist(lens)
    plt.title(t)
    plt.savefig(DATA_PATH+t+'_csc')
    plt.show()


def CrossVal_MAP(model,D):
    return Metrics.MAP(model.graph,D)


def ScoresGraphModels(graphs,graphs_titles,models,nbCascades,removed_pct_list):
    metrics = ['fit_time','MSE','MAP']
    data = {m:{str(g_t):dict() for g_t in graphs_titles} for m in metrics}
    curves = {m:{str(g_t):dict() for g_t in graphs_titles} for m in metrics}
    
    for g,t in zip(graphs,graphs_titles) : 
        cascades = csc.genCascades(g,nbCascades)
        saveCscadesDistrib(cascades,t)
        D = [csc.CascadeToTimeRepr(c) for c in cascades]
        
        for model in models: 
           res = evaluateModelCurve(model,t,g,cascades,removed_pct_list)
           for metric,r in zip(metrics,res):
               data[metric][t][str(model)] = r[0]
               curves[metric][t][str(model)] = r           

        data["MAP"][t]["original"] = Metrics.MAP(g,D)

    dfs = [pd.DataFrame(data[m]) for m in metrics]
    for df,m in zip(dfs,metrics):
        df.name = m
    return dfs,curves

def evaluateModel(model,gtitle,g,D,verbose= False):
    if (verbose):
        print("\nfor :",gtitle,str(model))
    crossVal_MSE = lambda m,d : Metrics.MSE(g, m.graph)
    model.set_params(csc.nodes_in_D(D))
    cross_res = cross_validate(model,D,scoring={"MAP":CrossVal_MAP,"MSE":crossVal_MSE})
    if(verbose):
        print(cross_res)
    time,MSE,MAP = cross_res['fit_time'].mean(),cross_res['test_MSE'].mean(),cross_res['test_MAP'].mean()
    return time,MSE,MAP

def evaluateModelCurve(model,gtitle,g,cascades,removed_pct_list):
    time = np.zeros(len(removed_pct_list))
    MSE = np.zeros(len(removed_pct_list))
    MAP = np.zeros(len(removed_pct_list))
    nodesInG = np.unique(sum(g.keys(),())).astype(int)
    for i,pct in enumerate(removed_pct_list):
        partial_cascades = csc.remove_Xpct_users(nodesInG,cascades,pct)
        D = [csc.CascadeToTimeRepr(c) for c in partial_cascades]
        time[i],MSE[i],MAP[i] = evaluateModel(model,gtitle,g,D,True)
    return time,MSE,MAP

def launch_test(nbCascades):
    # Generate GRAPHS 
    nx_to_dok_rand = lambda g : GGen.randomize_edges_values(
                                    dok_matrix(nx.to_scipy_sparse_matrix(g)))
    
    scale_free = nx.scale_free_graph(100)
    scale_free_dok = nx_to_dok_rand(scale_free)
    
    erdos_renyi = nx.erdos_renyi_graph(100,p=0.02)
    erdos_renyi_dok= nx_to_dok_rand(erdos_renyi)
    
    connected_cave_man = nx.connected_caveman_graph(12,6)
    connected_cave_man_dok= nx_to_dok_rand(connected_cave_man)
    
    bara = nx.barabasi_albert_graph(50,2)
    bara_dok = nx_to_dok_rand(bara)

    graphs =[scale_free,erdos_renyi,connected_cave_man,bara]
    graphs_dok = [scale_free_dok,erdos_renyi_dok,connected_cave_man_dok,bara_dok]
    graphs_titles = [f"scale_free", f"erdos_renyi",
                     f"connected_cave_man", f"barabasi"]
    
    removed_pct_list = np.linspace(0,0.8,9)
    # train and test 
    models=  [IC_EM_NotContiguous(),IC_EM_Saito2008()]
    dfs,curves = ScoresGraphModels(graphs_dok,graphs_titles,models,nbCascades,removed_pct_list)
    pd.set_option('display.max_columns', len(graphs))
    
    # Save network graphs 
    for g,dok,t in zip(graphs,graphs_dok,graphs_titles) : 
        plt.title(t+f"{dok.shape[0]}") 
        nx.draw_networkx(g,with_labels=False,node_size=30)
        plt.savefig(DATA_PATH+t+'_g')
        plt.show()
    
    # save perfs : 
    for df in dfs:
        print(f"\n{df.name}:\n{df}")
        with open(DATA_PATH+df.name+".md","w") as f:
            f.write(df.to_markdown())
            
    # save perf with missing data 
    for m_id,model in enumerate(models):
        for metric in ['MSE','MAP']:            
            plt.title(f'Perf for {str(model)}_{metric}')
            for g_id,t in enumerate(graphs_titles):
                c = curves[metric][t][str(model)]
                plt.plot(removed_pct_list,c,label=t)
            plt.legend()
            plt.xlabel('% of infections removed')
            plt.ylabel(f'{metric}_score')
            plt.savefig(f'{DATA_PATH}{str(model)}_{metric}.png')
            plt.show()
def main():
    print("start")
    start_time = time.time()
    launch_test(200)
    print("end " ,time.time()-start_time)
    

if __name__ == "__main__":
    main()