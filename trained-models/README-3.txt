cls v2.1
    - pretrained: 1st cnn layer
    - all encoder layers imitate alexnet backbone
    - set common seed (42)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs

cls v2.2
    - pretrained: 1st cnn layer
    - all encoder layers imitate alexnet backbone
    - added dropout after each encoder conv layer
    - set common seed (42)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs

cls v2.3
    - pretrained: 1st cnn layer
    - all encoder layers imitate alexnet backbone
    - added dropout after each encoder conv layer
    - *random init for all layers
    - set common seed (42)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs

grasp v2.1
    - pretrained: 1st cnn layer
    - all encoder layers imitate alexnet backbone
    - added dropout after each encoder conv layer
    - *random init for all layers
    - set common seed (42)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs