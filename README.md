# Model trainer for the NeuroVis project
This repo contains the essential code used to train our cls/grasp model using the Jacquard dataset (not included). Our cls/grasp model is a novel architecture that is capable of both classification and grasping tasks without any changes to network hyperparameters and training parameters. The following table summarizes the performance of our model in the two different tasks and the respective training specs:

 Task | CLS | Grasping
 :---: | :---: | :---:
 Accuracy (%) | 99+ | 73
 Epoch | 100 | 150
 Learning Rate | 2.5e^4 | 5e^4
 Loss | nll + bce (l1) | nll + l1

 ## Files to interact for training
 ```train.py``` - Train cls/grasp model (uncomment/comment specific lines to switch between different downstream tasks).

 ```test.py``` - Test model on testing dataset and visualize grasp / classification.

 ```data_preprocess.py``` - Convert original Jacqaurd dataset with .tiff, .png, .txt files (grasp candidates) to augmented dataset with only .npy files and grasp maps. Increases training speed significantly and obtains ground-truth labels suitable for our model architecture.
 
 ```parameters.py``` - Change parameters such as model name, learning rate, batchsize, etc.

