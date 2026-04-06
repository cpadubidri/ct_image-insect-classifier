import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join('/home/savvas/SUPER-NAS/USERS/Chirag/PROJECTS/202605-insect-ct', 'src')))

from datafeeder import get_dataloader
from train import train
from models import BinaryClassifier

from utils import Configuration



if __name__=="__main__":
    config_path = "experiments/exp_01/config.json"
    config = Configuration(config_path) 

    num_classes = 1
    model = BinaryClassifier(num_classes)

    
    train_loader, test_loader = get_dataloader(config)



    train(config=config, model=model, 
          train_loader=train_loader, 
          val_loader=test_loader)