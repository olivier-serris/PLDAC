# PLDAC

## Preliminary testing

###  Generated Graphs :

Les test sont faits sur les graphes suivants : <br>

![ScaleFree](IC_SAITO/data/eval/scale_free_g.png) ![ErdosRenyi](IC_SAITO/data/eval/erdos_renyi_g.png)  ![cave_man](IC_SAITO/data/eval/connected_cave_man_g.png)  ![Barabasi](IC_SAITO/data/eval/barabasi_g.png)

### Generated Cascades

Les longueurs des cascades ont les distributions suivantes : <br>

![ScaleFree](IC_SAITO/data/eval/scale_free_csc.png) ![ErdosRenyi](IC_SAITO/data/eval/erdos_renyi_csc.png)  ![cave_man](IC_SAITO/data/eval/connected_cave_man_csc.png)  ![Barabasi](IC_SAITO/data/eval/barabasi_csc.png)


### IC_Sait_EM testing

|  MEAN TIME               |   scale_free_100 |   erdos_renyi_50 |   connected_cave_man_40 |   barabasi_30 |
|:-------------------------|-----------------:|-----------------:|------------------------:|--------------:|
| IC_EM_NotContiguoust=0.1 |          1.40438 |          4.97177 |                2.29589  |       3.84879 |
| IC_EM_Saito2008_t=0.1    |          1.04466 |          1.48646 |                0.656524 |       1.45412 |

| CROSS-VAL MSE            |   scale_free_100 |   erdos_renyi_50 |   connected_cave_man_40 |   barabasi_30 |
|:-------------------------|-----------------:|-----------------:|------------------------:|--------------:|
| IC_EM_NotContiguoust=0.1 |          65.7175 |          16.1684 |                 22.1013 |       18.9839 |
| IC_EM_Saito2008_t=0.1    |          20.0648 |          11.4921 |                 21.7548 |       16.197  |

|    CROSS-VAL MAP         |   scale_free_100 |   erdos_renyi_50 |   connected_cave_man_40 |   barabasi_30 |
|:-------------------------|-----------------:|-----------------:|------------------------:|--------------:|
| IC_EM_NotContiguoust=0.1 |         0.905622 |         0.847066 |                0.90007  |      0.895651 |
| IC_EM_Saito2008_t=0.1    |         0.92538  |         0.851482 |                0.917274 |      0.896683 |
| original                 |         0.981216 |         0.913629 |                0.947582 |      0.936219 |

### Missing user testing

Courbes de performance selon le % d'infections retirée :

![NotContiguous_MSE](IC_SAITO/data/eval/_IC_EM_NotContiguous_MSE.png) ![NotContiguous_MAP](IC_SAITO/data/eval/_IC_EM_NotContiguous_MAP.png)  ![IC_EM_Saito2008_MSE](IC_SAITO/data/eval/_IC_EM_Saito2008_MSE.png)  ![IC_EM_Saito2008_MAP](IC_SAITO/data/eval/_IC_EM_Saito2008_MAP.png)
