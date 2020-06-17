
from scipy.sparse import dok_matrix, csr_matrix
import pprint as pp
import numpy as np
from EM_CTIC import EM_CTIC

from Sampling import CTIC_Cascade
import utils as utils



def handmade_graph_test():
    
    # Params 
    np.random.seed(2)
    np.set_printoptions(precision=2)
    maxT = 100
    names={0:"A",1:"B",2:"C",3:"D",4:"E"}
    k={(0,1):0.5,(0,2):0.5,(1,3) :0.5, (2,3):0.4, (2,4):0.6}
    r={(0,1):1,(0,2):2,(1,3) :2, (2,3):2, (2,4):2}
    graph=(names,k,r)

    # Gen cascades : 
    infections = np.array([(0,0)])
    times = np.array([CTIC_Cascade(graph,infections,0,maxT = maxT) for i in range(100)])

    # Test model 
    pp.pprint(times)
    nodes = list(names.keys())    
    model = EM_CTIC(nodes,maxT)
    predicted = model.EM(times)

    # Show results : 
    print("predicted : ")
    _,k_uv,r_uv = predicted 
    print("k_uv")
    pp.pprint(k_uv)
    print("r_uv")
    pp.pprint(r_uv)

def main():
    print("start")
    handmade_graph_test()
    print('end')

main()