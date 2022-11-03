cls v2.1
    - pretrained: 1st cnn layer
    - all encoder layers imitate alexnet backbone
    - set common seed (42)
    - lr: 5e-4
    - batchsize: 64
    - pretrained unfrozen after epoch 75
    - lrscheduler: x0.5 per 25 epochs