# Model trainer for the NeuroVis project
This repo contains the essential code used to train our cls/grasp model using the Jacquard dataset (not included). Our cls/grasp model is a novel architecture that is capable of both classification and grasping tasks without any changes to network hyperparameters and training parameters. The following table summarizes the performance of our model in the two different tasks and the respective training specs:

 Task | CLS | Grasping
 :---: | :---: | :---:
 Train Accuracy (%) | 87.00 | 76.10
 Test Accuracy (%) | 81.25 | 78.25
 Epoch | 150 | 150
 Learning Rate | 5e^4 | 5e^4
 Loss | double-log | double-log

 ## Files to interact for training
 ```train.py``` - Train cls/grasp model (uncomment/comment specific lines to switch between different downstream tasks).

 ```test.py``` - Test model on testing dataset and visualize grasp / classification.

 ```data_preprocess.py``` - Convert original Jacqaurd dataset with .tiff, .png, .txt files (grasp candidates) to augmented dataset with only .npy files and grasp maps. Increases training speed significantly and obtains ground-truth labels suitable for our model architecture.

 ```parameters.py``` - Change parameters such as model name, learning rate, batchsize, etc.

## Folder descriptions
```data``` - Location for training dataset.
Sample directory structure:
```
data
├── cls_top_5.txt
├── top_5
|   ├── train
|   |   ├── Chair
|   |   |   ├── 1a2a5a06ce083786581bb5a25b17bed6
|   |   |   |   ├── 4_1a2a5a06ce083786581bb5a25b17bed6_0_grasps.txt
|   |   |   |   ├── 4_1a2a5a06ce083786581bb5a25b17bed6_grasps.txt
|   |   |   |   ├── 4_1a2a5a06ce083786581bb5a25b17bed6_mask.png
|   |   |   |   ├── 4_1a2a5a06ce083786581bb5a25b17bed6_perfect_depth.tiff
|   |   |   |   ├── 4_1a2a5a06ce083786581bb5a25b17bed6_RGB.png
|   |   |   |   ├── 4_1a2a5a06ce083786581bb5a25b17bed6_stereo_depth.tiff
|   |   |   ├── 1b7bef12c554c1244c686b8271245d1b
|   |   ├── figurines
|   |   ├── Lamp
|   |   ├── pen+pencil
|   |   ├── plants
|   ├── test
|   |   ├── ...
```

```inference``` - Contains all architectures of models used during testing. Contains code referenced from [robitc-grasping](https://github.com/skumra/robotic-grasping) repo.

```logs``` - Saves training log

```trained-models``` - Saves trained models (checkpoints + final model)

```vis``` - Saves network output visualizations.

This work is under the MIT Liscence
