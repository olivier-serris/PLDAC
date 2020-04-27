# PLDAC

## Preliminary testing

###  Generated Graphs :

Les test sont faits sur les graphes suivants : <br>

![ScaleFree](IC_SAITO/data/eval/scale_free_g.png) ![ErdosRenyi](IC_SAITO/data/eval/erdos_renyi_g.png)  ![cave_man](IC_SAITO/data/eval/connected_cave_man_g.png)  ![Barabasi](IC_SAITO/data/eval/barabasi_g.png)

### Generated Cascades

Les longueurs des cascades ont les distributions suivantes : <br>

![ScaleFree](IC_SAITO/data/eval/scale_free_csc.png) ![ErdosRenyi](IC_SAITO/data/eval/erdos_renyi_csc.png)  ![cave_man](IC_SAITO/data/eval/connected_cave_man_csc.png)  ![Barabasi](IC_SAITO/data/eval/barabasi_csc.png)


### IC_Sait_EM testing

|          MSE        |   scale_free |   erdos_renyi |   connected_cave_man |   barabasi |
|:--------------------|-------------:|--------------:|---------------------:|-----------:|
| IC_EM_Saito2008     |   0.00172863 |    0.0018261  |           0.00520663 |  0.0142106 |
| IC_EM_NotContiguous |   0.00471561 |    0.00741874 |           0.0148063  |  0.153406  |

|           MAP       |   scale_free |   erdos_renyi |   connected_cave_man |   barabasi |
|:--------------------|-------------:|--------------:|---------------------:|-----------:|
| IC_EM_Saito2008     |     0.982853 |      0.849562 |             0.840047 |   0.891198 |
| IC_EM_NotContiguous |     0.979134 |      0.656865 |             0.726627 |   0.71513  |
| original            |     0.980544 |      0.905014 |             0.933889 |   0.951407 |

### Missing user testing

Courbes de performance selon le % d'infections retirée :

|  Missing Users   |                               MSE                                   |                                   MAP                              |
|:-----------------|--------------------------------------------------------------------:|-------------------------------------------------------------------:|
| EM_NotContiguous |![NotContiguous_MSE](IC_SAITO/data/eval/IC_EM_NotContiguous_MSE.png) |![NotContiguous_MAP](IC_SAITO/data/eval/IC_EM_NotContiguous_MAP.png)|
| EM_Saito         |![IC_EM_Saito2008_MSE](IC_SAITO/data/eval/IC_EM_Saito2008_MSE.png)   |![IC_EM_Saito2008_MAP](IC_SAITO/data/eval/IC_EM_Saito2008_MAP.png)  |
