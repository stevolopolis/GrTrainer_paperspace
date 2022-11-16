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
    - added dropout after each encoder conv layer (0.2)
    - set common seed (42)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs

cls v2.3
    - pretrained: 1st cnn layer
    - all encoder layers imitate alexnet backbone
    - added dropout after each encoder conv layer (0.2)
    - *random init for all layers
    - set common seed (42)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs

cls v3.1
    - pretrained: 1st cnn layer
    - beginning channel count: 64
    - added dropout after each encoder conv layer (0.3)
    - *random init for all layers
    - set common seed (42)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs

cls v3.2
    - pretrained: 1st cnn layer
    - beginning channel count: 32
    - added dropout after each encoder conv layer (0.3)
    - *random init for all layers
    - set common seed (42)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs

cls v2.10.1
    - pretrained: 1st cnn layer
    - beginning channel count: 32
    - added dropout after each encoder conv layer (0.3)
    - *random init for all layers
    - set common seed (42)
    - feature-distillation with (0.5 - scheduler starting from 0.2)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs

cls v2.10.2
    - pretrained: 1st cnn layer
    - beginning channel count: 32
    - added dropout after each encoder conv layer (0.3)
    - *random init for all layers
    - set common seed (42)
    - feature-distillation with (0.5 - scheduler starting from 1.0)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs

cls v2.10.3
    - pretrained: 1st cnn layer
    - beginning channel count: 32
    - added dropout after each encoder conv layer (0.3)
    - *random init for all layers
    - set common seed (42)
    - feature-distillation with (0.5 - scheduler starting from 1.0; first distill then train)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs

cls v2.11.1
    - pretrained: 1st cnn layer
    - beginning channel count: 32
    - added dropout after each encoder conv layer (0.3)
    - *random init for all layers
    - set common seed (42)
    - feature-distillation with (0.5 - schedule starting from 1.0; distillLoss + Loss)
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

grasp v2.11.1
    - pretrained: 1st cnn layer
    - beginning channel count: 32
    - added dropout after each encoder conv layer (0.3)
    - *random init for all layers
    - set common seed (42)
    - feature-distillation with (0.5 - schedule starting from 1.0; distillLoss + Loss)
    - lr: 5e-4
    - batchsize: 128
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs