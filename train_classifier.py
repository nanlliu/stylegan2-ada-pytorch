import torch
import torch.nn
import numpy as np

import time
import copy
import json
import PIL
import os

from torch.utils.data import Dataset, DataLoader

import legacy
import imageio

from projector import project

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ClevrDataset(Dataset):
    def __init__(self, json_data_path):
        self.data = json.load(json_data_path)
    
    def __getitem__(self, index):
        im_name = self.data[index][0]
        label = self.data[index][1]
        return im_name, label
    
    def __len__(self):
        return self.data.shape[0]


def train_model(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, start_epoch=0, num_epochs=50):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    output_splits = torch.split(outputs, split_size_or_sections=19, dim=1)
                    
                    # calculate loss
                    loss = 0
                    preds = []
                    for i, split in enumerate(output_splits):
                        # split represent each object
                        # split = [N, 19] 4 shapes + 3 sizes + 9 colors + 3 materials
                        attr_splits = torch.split(split, split_size_or_sections=[4, 3, 9, 3], dim=1)
                        for j, attr_split in enumerate(attr_splits):
                            # attr_split = [N, M=(4,3,9,3)]
                            # iterate each attribute and calculate loss
                            _, curr_preds = torch.max(attr_split, 1)
                            loss += criterion(attr_split, labels[:, i * 4 + j])
                            preds.append(curr_preds)
                    
                    # 20 attributes for 5 objects
                    loss /= 20
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    preds = torch.stack(preds, dim=1)
                    running_loss += loss.item() * inputs.size(0)
                    sum_rows = torch.sum(preds == labels.data, dim=1)
                    running_corrects += torch.sum(sum_rows == 20)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                },
                    f'/root/stylegan2-ada-pytorch/checkpoints/{epoch + 1}.tar')
        
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    pass