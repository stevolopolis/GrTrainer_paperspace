"""Thie file contains the main code for training the relevant networks.

Available models for training include:
- CLS model (with/without imagenet pretraining)
|   - grconvnet
|   - alexnet
- Grasp model (with/withou imagenet pretraining)
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
from torch.optim import Adam, SGD

from tqdm import tqdm

import inference.models.alexnet as models
from paths import Path
from parameters import Params
from data_loader import DataLoader
from utils import epoch_logger, log_writer, get_correct_preds, get_acc
from grasp_utils import get_correct_grasp_preds
from evaluation import get_test_acc
from loss import minLossTarget

params = Params() 
paths = Path()

# Create <trained-models> directory
paths.create_model_path()
# Create directory for training logs
paths.create_log_path()
# Create subdirectory in <logs> for current model
paths.create_model_log_path()

# Load model
model = models.AlexnetMap(n_cls=params.NUM_CLASS).to(params.DEVICE)

# Load checkpoint weights
checkpoint_name = 'alexnetGrasp_depthconcat_convtrans_top5_v4.3'
checkpoint_epoch = 50
#checkpoint_path = os.path.join(params.MODEL_PATH, checkpoint_name, '%s_epoch%s.pth' % (checkpoint_name, checkpoint_epoch))
#model.load_state_dict(torch.load(checkpoint_path))

# Create DataLoader class
data_loader = DataLoader(params.TRAIN_PATH, params.BATCH_SIZE, params.TRAIN_VAL_SPLIT)
# Get number of training/validation steps
n_train, n_val = data_loader.get_train_val()

# Training utils
optim = SGD(model.parameters(), lr=params.LR, momentum=.9)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 25, 0.1)
# Loss fn for CLS without final activation
criterion = nn.CrossEntropyLoss()
# Loss fn for Grasping
grasp_criterion = nn.MSELoss()

for epoch in tqdm(range(1, params.EPOCHS + 1)):
    if epoch == 15:
        model.unfreeze_depth_backbone()

    train_history = []
    val_history = []
    train_total = 0
    train_correct = 0
    val_total = 0
    val_correct = 0
    # Data loop for CLS training
    #for step, (img, label) in enumerate(data_loader.load_batch()):
    # Data loop for Grasp training
    for step, (img, label, candidates) in enumerate(data_loader.load_grasp_batch()):
        optim.zero_grad()
        output = model(img)

        # Loss fn and data processing for CLS training
        #loss = criterion(output, label)
        #label = F.one_hot(label, num_classes=params.NUM_CLASS)
        #label = label.type(torch.float32)
        # Loss fn for Grasp training
        if epoch >= 15:
            label = minLossTarget(output, candidates).to(params.DEVICE)
        loss = grasp_criterion(output, label)

        if step < n_train:
            loss.backward()
            optim.step()

            # Write loss to log file -- 'logs/<model_name>/<model_name>_log.txt'
            log_writer(params.MODEL_NAME, epoch, step, loss.item(), train=True)
            train_history.append(loss.item())
            # Correct prediction stats for CLS training
            #correct, total = get_correct_preds(output, label)
            # Correct prediction stats for Grasp traning
            correct, total = get_correct_grasp_preds(output, candidates)
            train_correct += correct
            train_total += total
        else:
            log_writer(params.MODEL_NAME, epoch, step, loss.item(), train=False)
            val_history.append(loss.item())
            # Correct prediction stats for CLS training
            #correct, total = get_correct_preds(output, label)
            # Correct prediction stats for Grasp traning
            correct, total = get_correct_grasp_preds(output, candidates)
            val_correct += correct
            val_total += total
    
    # Get testing accuracy stats
    test_acc, test_loss = 0, 0  # get_test_acc(model)
    scheduler.step()

    # Get training and validation accuracies
    train_acc = get_acc(train_correct, train_total)
    val_acc = get_acc(val_correct, val_total)
    # Write epoch loss stats to log file
    epoch_logger(params.MODEL_NAME, epoch, train_history, val_history, test_loss, train_acc, val_acc, test_acc)
    # Save checkpoint model -- 'trained-models/<model_name>/<model_name>_epoch<epoch>.pth'
    torch.save(model.state_dict(), os.path.join(params.MODEL_LOG_PATH, f"{params.MODEL_NAME}_epoch{epoch}.pth"))

# Save final epoch model
torch.save(model.state_dict(), os.path.join(params.MODEL_LOG_PATH, f"{params.MODEL_NAME}_final.pth"))