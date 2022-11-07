cls_v1 / grasp_v1
	- unfreeze (epoch25)
	- lr_scheduler (per 50 epoch)

cls_v2 (FAIL)
	- unfreeze (epoch75)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d only after encoder convs (before relu)
	- height reranged to [-1, 1]
	- old logl1loss
	- background label = [-1, -1, -1, -1, -1, -1]

cls_v3 (FAIL)
	- unfreeze (epoch75)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d only after encoder convs (before relu)
	- height reranged to [-1, 1]
	- old logl1loss
	- background label = [-1, -1, -1, -1, -1, 0.0]

cls_v4 (FAIL)
	- unfreeze (epoch75)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d only after encoder convs (before relu)
	- height reranged to [-1, 1]
	- old logl1loss
	- background label = [-1, -1, -1, -1, -1, 0.0]

cls_v5 (SUCCESS - 75.25% test acc)
	- unfreeze (epoch75)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d only after encoder convs (before relu)
	- height reranged to [-1, 1]
	- old logl1loss
	- background label = [-1, -1, -1, -1, -1, 0.0]

cls_v6 (SUCCESS - 78.25% test acc)
	- unfreeze (epoch75)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d only after encoder convs (before relu)
	- res-block after encoder conv (x2)
	- dropout after each encoder conv
	- height reranged to [-1, 1]
	- old logl1loss
	- background label = [-1, -1, -1, -1, -1, 0.0]

cls_v7 
	- unfreeze (epoch75)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d only after encoder convs (before relu)
	- res-block after encoder conv (x2)
	- dropout after each encoder conv AND res-block
	- height reranged to [-1, 1]
	- old logl1loss
	- background label = [-1, -1, -1, -1, -1, 0.0]


grasp_v2
	- unfreeze (epoch50)
	- lr_scheduler (per 25 epoch)
	- dropout(0.2) per conv layer (not at deconvs)

grasp_v3
	- unfreeze (epoch50)
	- lr_scheduler (per 25 epoch)
	- no dropout

grasp_v4
	- unfreeze (epoch50)
	- lr_scheduler (per 25 epoch)
	- no dropout
	- newLoss -> double step-wise

grasp_v5
	- unfreeze (epoch50)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d after each conv before each relu
	- old logl1loss

grasp_v6 (trained till epoch64 then disconnected)
	- unfreeze (epoch50)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d only after encoder convs (before relu)
	- old logl1loss

grasp_v7
	- unfreeze (epoch75)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d only after encoder convs (before relu)
	- old logl1loss

grasp_v8
	- unfreeze (epoch75)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d after convs (before relu)
	- resnet blocks x3 in the middle
	- old logl1loss

grasp_v9
	- unfreeze (epoch75)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d only after encoder convs (before relu)
	- resnet block x3 in the middle
	- old logl1loss

grasp_v10 (SUCCESS - 71% test acc)
	- unfreeze (epoch75)
	- lr_scheduler (per 25 epoch)
	- batchnorm2d only after encoder convs (before relu)
	- height reranged to [-1, 1]
	- old logl1loss