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

 ```clip.py``` - Zero-shot labelling using OpenAI CLIP model. Achieves ~85+ accuracy on our top_5 jacquard dataset.

## Additional note to zero-shot labelling
Current clip model and label group (top_5) achieves 85+% accuracy. Here are some recommended next-steps to increase the accuracy and to incorporate cheap manual labelling:
1. Experiment with other vocabs that suit the Jacquard objects better (use cnf matrix as reference to which objects are more difficult for zero-shot classification)
2. (Human inspection) To expand to entire Jacquard dataset, we first need to curate a dictionary of the class labels
3. (Human inspection) Group all images of same predicted class. Open in file exporer and set all images to "BIG ICON". That way, outliers could be quickly identified and relablled.

## .py file descriptions
- **For data**
    - `data_loader_v2.py` - **major** data_loader class for training our models
    - `data_preprocess.py` - to convert grasp candidate lists into grasp maps where each pixel has a grasp candidate. Also contains functons for converting images into .npy files for faster inferencing and other basic image processing functions.
    - `data_postprocess.py` - sanity check for `data_preprocess.py` by reverting the function calls and visualizing the classifications/grasps
- **For training**
    - `train.py` - main training code for both GRASP and CLS model. Uncomment cerrtain lines of code to swap between GRASP and CLS training.
    - `parameters.py` - contains all the relevant paths and hparams required for training
    - `loss.py` - contains special loss functions such as double-log loss and other losses that were experimented but suboptimal (can ignore)
    - `utils.py` - contains some miscellaneous functions such as training logger and accuracy calculator
    - `grasp_utils.py` - contains miscellaneous functions specific for grasping network (taken from GrConvnet Repo)
    - `model_utils.py` - contains functions for hooking model's hidden layer outputs (for RSM analysis)
- **For testing**
    - `test.py`
    - `evaluation.py`
- **For representational similarity analysis (RSM)**
    - `rsm_deepModelAnalysis.py`
    - `rsm_generate_SSM.py`
    - `rsm_compare_reps.py`
- **For neuron shapley simulation**
    - `shapley_cb_aggregate.py`
    - `shapley_cb_run.py`

## Folder descriptions
```inference``` - Contains all architectures of models used during testing. Contains code referenced from [robotic-grasping](https://github.com/skumra/robotic-grasping) repo.

```shap``` - Shapley values for selected layers the best models

```guided_backprop_summaries``` - selected images that explain certain observations we made when comparing the behaviors of CLS vs GRASP models

```logs``` - Saves training log

```trained-models``` - Saves trained models (checkpoints + final model)

```vis``` - Saves network output visualizations.
- ```<model_name>_<cls/grasp>_v<version>_epoch<epoch>``` - contains visualizations of CLS/GRASP model predictions
- ```am``` - contains early-layer AM visualizations of the best CLS/GRASP model
- ```inter_model_corr``` - contains RSMs that compare layer activations across different models
- ```intralayer_corr``` - contains RSMs that compare layer activations within a model
- ```misc_MAIN_images``` - miscellaneous images used for the MAIN conference
- ```shap``` - contains images of shapley confidence bounds and shapley distributions that progresses over simulation iterations

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
