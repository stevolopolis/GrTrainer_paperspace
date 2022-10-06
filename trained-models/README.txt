alexnetGrasp_map_top5_v5.2.2 (BEST - ~78)
|	- bbox_loss * 2 + conf_loss * 1
|	- 5e-4


LR: 2.5e-4
batch_size: 128
epoch: ~100

(~90+ acc; cls)
cls_v100
- nll + bce (epoch100)
(73% acc; grasp)
grasp_v110
- nll + l1 * 2 (with rotation; LR=5e-4; epoch150)

~55-66% acc (grasp)

grasp_v100
- nll + l2
grasp_v101
- nll + l2 * 2
grasp_v102
- l2 + l2 * 2
grasp_v103
- l1 + l1 * 2
grasp_v104
- nll + l2 * 2 (with rotation)
grasp_v105
- nll + l1 * 2 (with rotation)
grasp_v106
- nll + l2 * 2 (with rotation) 
grasp_v106
- nll + l2 * 2 (with rotation; LR=1e-3)
grasp_v107
- nll + l2 * 3 (with rotation; LR=2.5e-4)
grasp_v109
- nll + l2 * 2 (with rotation; LR=5e-4)

-------
grasp_v110
- nll + l1 * 2 (with rotation; LR=5e-4)