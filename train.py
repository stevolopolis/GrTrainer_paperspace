"""Thie file contains the main code for training the relevant networks.

Available models for training include:
- CLS model (with/without imagenet pretraining)
|   - grconvnet
|   - alexnet
- Grasp model (with/without imagenet pretraining)
|   - alexnet

Comment or uncomment certain lines of code for swapping between
training CLS model and Grasping model.

E.g. Uncomment the lines with NO SPACE between '#' and the codes: 
"Training for Grasping"
# Loss fn for CLS training
#loss = nn.CrossEntropyLoss()
# Loss fn for Grasping
loss = nn.MSELoss()

----->

# Loss fn for CLS training
loss = nn.CrossEntropyLoss()
# Loss fn for Grasping
#loss = nn.MSELoss()

"""

import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.models import alexnet

from tqdm import tqdm

import inference.models.alexnet as models
import inference.models.grConvMap as modelsGr
from paths import Path
from parameters import Params
from data_loader_v2 import DataLoader
from utils import epoch_logger, log_writer, get_correct_cls_preds_from_map, get_acc
from grasp_utils import get_correct_grasp_preds_from_map
from evaluation import get_cls_acc, get_grasp_acc
from loss import MapLoss, DistillationLoss

SEED=42
params = Params() 
paths = Path()

# Create <trained-models> directory
paths.create_model_path()
# Create directory for training logs
paths.create_log_path()
# Create subdirectory in <logs> for current model
paths.create_model_log_path()

# Set common seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# Load model
model = models.AlexnetMap_v3().to(params.DEVICE)
#model = modelsGr.GrConvMap().to(params.DEVICE)

# Teacher model
pretrained_alexnet = alexnet(pretrained=True).to(params.DEVICE)
pretrained_alexnet.eval()
for weights in pretrained_alexnet.features.parameters():
    weights.requires_grad = False

# Load checkpoint weights
checkpoint_name = 'alexnetGrasp_depthconcat_convtrans_top5_v4.3'
checkpoin_epoch = 50
#checkpoint_path = os.path.join(params.MODEL_PATH, checkpoint_name, '%s_epoch%s.pth' % (checkpoint_name, checkpoint_epoch))
#model.load_state_dict(torch.load(checkpoint_path))

# Create DataLoader class
data_loader = DataLoader(params.TRAIN_PATH, params.BATCH_SIZE, params.TRAIN_VAL_SPLIT, seed=SEED)
# Get number of training/validation steps
n_train, n_val = data_loader.get_train_val()

# Training utils
optim = Adam(model.parameters(), lr=params.LR)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 25, 0.5)

for epoch in tqdm(range(1, params.EPOCHS + 1)):
    if epoch == 75:
        model.unfreeze_depth_backbone()

    train_history = []
    val_history = []
    train_total = 1
    train_correct = 1
    val_total = 1
    val_correct = 1
    # Data loop for CLS training
    #for step, (img, map, label) in enumerate(data_loader.load_batch()):
    # Data loop for Grasp training
    for step, (img, map, label) in enumerate(data_loader.load_grasp_batch()):
        optim.zero_grad()
        output = model(img)

        # Loss fn for CLS/Grasp training
        loss = MapLoss(output, map)
        # Distillation loss (experimental)
        #distill_loss = DistillationLoss(img, model, pretrained_alexnet, model_s_type='alexnetMap', model_t_type='alexnet')
        #loss = loss + distill_loss * params.DISTILL_ALPHA
        
        if step < n_train:
            loss.backward()
            optim.step()

            # Write loss to log file -- 'logs/<model_name>/<model_name>_log.txt'
            log_writer(params.MODEL_NAME, epoch, step, loss.item(), train=True)
            train_history.append(loss.item())
            # Dummie prediction stats
            correct, total = 0, 1
            train_correct += correct
            train_total += total
        else:
            log_writer(params.MODEL_NAME, epoch, step, loss.item(), train=False)
            val_history.append(loss.item())
            # Dummie prediction stats
            correct, total = 0, 1
            val_correct += correct
            val_total += total
 
    # Get testing accuracy stats (CLS / Grasp)
    if (epoch % 10 == 1):
        model.eval()
        #train_acc, train_loss = get_cls_acc(model, include_depth=True, seed=SEED, dataset=params.TRAIN_PATH, truncation=None)
        train_acc, train_loss = get_grasp_acc(model, include_depth=True, seed=SEED, dataset=params.TRAIN_PATH, truncation=None)
        #test_acc, test_loss = get_cls_acc(model, include_depth=True, seed=SEED, dataset=params.TEST_PATH, truncation=None)
        test_acc, test_loss = get_grasp_acc(model, include_depth=True, seed=SEED, dataset=params.TRAIN_PATH, truncation=None)
        scheduler.step()

        # Experimental
        #params.DISTILL_ALPHA /= 2
        
        model.train()

    # Get training and validation accuracies
    val_acc = train_acc # get_acc(val_correct, val_total)
    # Write epoch loss stats to log file
    epoch_logger(params.MODEL_NAME, epoch, train_history, val_history, test_loss, train_acc, val_acc, test_acc)
    # Save checkpoint model -- 'trained-models/<model_name>/<model_name>_epoch<epoch>.pth'
    torch.save(model.state_dict(), os.path.join(params.MODEL_LOG_PATH, f"{params.MODEL_NAME}_epoch{epoch}.pth"))

# Save final epoch model
torch.save(model.state_dict(), os.path.join(params.MODEL_LOG_PATH, f"{params.MODEL_NAME}_final.pth"))