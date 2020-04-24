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
from functools import partial

DATA_PATH = '../data/eval/'

def handmadeGraphTest():
    '''' Test de performance avec graph fait main '''
    handmadeGraph=dok_matrix((5,5),dtype=np.float32)
    handmadeGraph[0,1] =0.5
    handmadeGraph[0,2] =0.5
    handmadeGraph[1,3] =0.6
    handmadeGraph[2,3] =0.4
    handmadeGraph[2,4] =0.6
    
    cascades = csc.genCascades(handmadeGraph,50)
    D = [csc.CascadeToTimeRepr(c) for c in cascades]
    CrossVal_MAP = lambda model,D : Metrics.MAP(model.graph,D)

    models= [IC_EM_Saito2008(),IC_EM_NotContiguous()]
    scores = dict()
    for i,model in enumerate(models):
        model.set_params(csc.nodes_in_D(D))
        cross_res = cross_validate(model,D,scoring=CrossVal_MAP)
        scores[str(model)] = cross_res['test_score'].mean()
    print("score for handmadegraph")
    scores["original"] = Metrics.MAP(handmadeGraph,D)
    pp.pprint(scores)

def saveCscadesDistrib(Cascades,t):
    '''Sauvegarde la distribution de longeur de cascades sous forme d'histogramme'''
    lens = np.array([len(c) for c in Cascades])
    plt.hist(lens)
    plt.title(t)
    plt.savefig(DATA_PATH+t+'_csc')
    plt.show()

def ScoresGraphModels(graph_dict,models,metrics,nbCascades,removed_pct_list):
    indicators = list(metrics.keys())+['fit_time']
    data = {m:{str(g_t):dict() for g_t in graph_dict.keys()} for m in indicators}
    curves = {m:{str(g_t):dict() for g_t in graph_dict.keys()} for m in indicators}
    
    for gtitle,g in graph_dict.items() : 
        cascades = csc.genCascades(g,nbCascades)
        saveCscadesDistrib(cascades,gtitle)
        
        nodesInG = np.unique(sum(g.keys(),())).astype(int)
        partial_cascades = [csc.remove_Xpct_users(nodesInG,cascades,pct) 
                            for pct in removed_pct_list]
        
        for model in models: 
            print("\nfor :",gtitle,str(model))
            res = evaluateModelCurve(model,g,cascades,metrics,partial_cascades)
            for metric,r in res.items():
                data[metric][gtitle][str(model)] = r[0]
                curves[metric][gtitle][str(model)] = r
        if ("MAP" in metrics.keys()):
            D = [csc.CascadeToTimeRepr(c) for c in cascades]
            data["MAP"][gtitle]["original"] = Metrics.MAP(g,D)
    print(f"data : {data}")
    dfs = [pd.DataFrame(data[m]) for m in indicators]
    for df,m in zip(dfs,indicators):
        df.columns.name = m
    return dfs,curves

def evaluateModel(model,g,D,metrics):
    ''' return res of cross validation for model g wit data D and metrics
        the result is a dict  {metric_name : metric_result}'''
    scoring_metrics = {m : partial(metrics[m],g=g) for m in metrics.keys() }
    g = None    
    model.set_params(csc.nodes_in_D(D))
    cross_res = cross_validate(model,D,scoring=scoring_metrics)
    print(f'\n{cross_res}')
    scores = {m:cross_res[f'test_{m}'].mean() for m in metrics.keys()}
    scores['fit_time'] = cross_res['fit_time'].mean()
    return scores

def evaluateModelCurve(model,g,cascades,metrics,partial_cascades):
    data = {m:np.zeros(len(partial_cascades)) for m in metrics.keys()}
    data['fit_time'] = np.zeros(len(partial_cascades))
    for i,partial_cascade in enumerate(partial_cascades):
        D = [csc.CascadeToTimeRepr(c) for c in partial_cascade]
        cross_val_dict = evaluateModel(model,g,D,metrics)
        for (metric,res) in cross_val_dict.items():
            data[metric][i] = res
    return data


def generateGraphs():
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
    graphs_dok = [scale_free_dok, erdos_renyi_dok,
                  connected_cave_man_dok, bara_dok]
    
    graphs_titles = [f"scale_free",  f"erdos_renyi",
                     f"connected_cave_man", f"barabasi"]
    # Save network graphs 
    for g,dok,t in zip(graphs,graphs_dok,graphs_titles) : 
        plt.title(t+f"{dok.shape[0]}") 
        nx.draw_networkx(g,with_labels=False,node_size=30)
        plt.savefig(DATA_PATH+t+'_g')
        plt.show()
    
    return dict(zip(graphs_titles,graphs_dok))

def launch_test(models,graph_dict,nbCascades,metrics):

    removed_pct_list = np.linspace(0,0.8,9)
    # train and test     
    dfs,curves = ScoresGraphModels(graph_dict,models,metrics,nbCascades,removed_pct_list)
        
    # save perfs :
    pd.set_option('display.max_columns', len(graph_dict))
    for df in dfs:
        print(f"perf :\n{df}")
        with open(DATA_PATH+df.columns.name+".md","w") as f:
            f.write(df.to_markdown())
            
    # save perf curves with missing data 
    for model in models:
        for metric in metrics.keys():            
            plt.figure()
            plt.title(f'Perf for {str(model)}_{metric}')
            for gtitle in graph_dict.keys():
                c = curves[metric][gtitle][str(model)]
                plt.plot(removed_pct_list,c,label=gtitle)
            plt.legend()
            plt.xlabel('% of infections removed')
            plt.ylabel(f'{metric}_score')
            plt.savefig(f'{DATA_PATH}{str(model)}_{metric}.png')
            plt.show()
            
def main():
    print("start : ")
    
    start_time = time.time()
    metrics = { 'MSE':(lambda model,D,g : Metrics.MSE(g, model.graph)),
                #'MAE':(lambda model,D,g : Metrics.MAE(g, model.graph)),
                'MAP':(lambda model,D,g : Metrics.MAP(model.graph,D)),
               }
    models=  [IC_EM_Saito2008(),
              IC_EM_NotContiguous()]
    graph_dict = generateGraphs()
    launch_test(models,graph_dict,200,metrics)
   
    print("End \nTest duration : " ,time.time()-start_time)
    
if __name__ == "__main__":
    main()