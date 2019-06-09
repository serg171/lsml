import argparse
import numpy as np
import os
import pandas as pd
from tensorboardX import SummaryWriter
import torch
import torchvision
from torch import nn
import json
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

import train
import train_with_Focalloss
import utils
import Dataset
import FocalLoss


#from torch import nn, optim
#from torch.optim import lr_scheduler
#import torch.nn.functional as F
#from collections import OrderedDict

def _parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--datadir', default='/data/iNat')
    parser.add_argument('--outpath', default='/data/iNat/runs/')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--csv_name', default='')
    return parser.parse_args()


def main(args):
    #np.random.seed(432)
    #torch.random.manual_seed(432)
    try:
        os.makedirs(args.outpath)
    except OSError:
        pass
    experiment_path = utils.get_new_model_path(args.outpath)

    train_writer = SummaryWriter(os.path.join(experiment_path, 'train_logs'))
    val_writer = SummaryWriter(os.path.join(experiment_path, 'val_logs'))
    trainer = train_with_Focalloss.Trainer(train_writer, val_writer)
    
    # making dataframes with file names and true answers
    # train
    ann_file = '/data/iNat/train2019.json'
    with open(ann_file) as data_file:
        train_anns = json.load(data_file)
    
    train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id','category_id']]
    train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
    df_train_file_cat = pd.merge(train_img_df, train_anns_df, on='image_id')
    
    # valid
    valid_ann_file = '/data/iNat//val2019.json'
    with open(valid_ann_file) as data_file:
        valid_anns = json.load(data_file)
           
    valid_anns_df = pd.DataFrame(valid_anns['annotations'])[['image_id','category_id']]
    valid_img_df = pd.DataFrame(valid_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
    df_valid_file_cat = pd.merge(valid_img_df, valid_anns_df, on='image_id')    
        
    # test
    ann_file = '/data/iNat/test2019.json'
    with open(ann_file) as data_file:
        test_anns = json.load(data_file)
    
    test_img_df = pd.DataFrame(test_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
    
    # make dataloaders
    ID_train = df_train_file_cat.file_name.values
    labels_train = df_train_file_cat.category_id.values
    training_set = Dataset.Dataset(ID_train, labels_train, root = '/data/iNat/train_val/') # train

    ID_test = df_valid_file_cat.file_name.values
    labels_test = df_valid_file_cat.category_id.values
    test_set = Dataset.Dataset(ID_test, labels_test, root = '/data/iNat/train_val/') # valid
    
    ID_to_sent = test_img_df.file_name.values
    to_sent_dataset = Dataset.Dataset_to_sent(ID_to_sent, root = '/data/iNat/test/')

    trainloader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=15)
    evalloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=15)
    to_sent_loader = DataLoader(to_sent_dataset, batch_size=args.batch_size, shuffle=False, num_workers=15)
    
    # pretrained model
    resnet = torchvision.models.resnet18(pretrained=True)
    classifier = nn.Linear(512, 1010)
    
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = classifier
    
    #densnet = torchvision.models.densenet121(pretrained=True)
    #classifier = nn.Linear(1024, 1010)
    #for param in densnet.parameters():
    #    param.requires_grad = False

    #densnet.classifier = classifier 
    
    model = resnet
    #model = densnet
    opt = torch.optim.SGD(model.parameters(), lr=0.0003)
    #opt = torch.optim.Adadelta(model.parameters())
    Loss_func = FocalLoss.FocalLoss(gamma=2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = args.epochs-1)
    
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        if epoch == args.epochs-1:
            for param in model.parameters():
                param.requires_grad = True
        
        trainer.train_epoch(model, opt, trainloader, schedule, Loss_func)
        metrics = trainer.eval_epoch(model, evalloader, Loss_func)

        state = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=opt.state_dict(),
            loss=metrics['loss'],
            accuracy=metrics['acc'],
            global_step=trainer.global_step,
        )
        export_path = os.path.join(experiment_path, 'last.pth')
        torch.save(state, export_path)
        print('Loss', metrics['loss'])
        print('Accuracy', metrics['acc'])
    
    # save predictions to csv
    utils.make_prediction(model, to_sent_loader, test_img_df.image_id.values, submit_name=args.csv_name)
             
if __name__ == "__main__":
    args = _parse_args()
    main(args)