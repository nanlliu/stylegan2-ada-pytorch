import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet152

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ClevrDataset(Dataset):
    def __init__(self, split):
        self.train_path = '/home/nan/data/relational_data_multi_objects_50000_train.npz'
        self.val_path = '/home/nan/data/relational_data_multi_objects_10000_val.npz'

        if split == 'train':
            self.data = np.load(self.train_path)
        elif split == 'val':
            self.data = np.load(self.val_path)
        else:
            raise NotImplemented

        self.ims = self.data['ims']
        self.labels = self.data['labels']
    
    def __getitem__(self, index):
        im = self.ims[index]
        label = self.labels[index]
        return im, label
    
    def __len__(self):
        return self.ims.shape[0]


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
                inputs = inputs.permute(0, 3, 1, 2).float().to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    output_splits = torch.split(outputs, split_size_or_sections=[22, 22, 7], dim=1)

                    # calculate loss
                    loss = 0
                    preds = []

                    object1_attr_splits = torch.split(output_splits[0], split_size_or_sections=[4, 3, 9, 3, 3], dim=1)
                    object2_attr_splits = torch.split(output_splits[1], split_size_or_sections=[4, 3, 9, 3, 3], dim=1)
                    relation_attr = output_splits[2]

                    for i, object_attr in enumerate([object1_attr_splits, object2_attr_splits]):
                        for j, attr in enumerate(object_attr):
                            _, curr_preds = torch.max(attr, 1)
                            loss += criterion(attr, labels[:, i * 5 + j])
                            preds.append(curr_preds)

                    # relation preditions
                    _, rel_preds = torch.max(relation_attr, dim=1)
                    loss += criterion(relation_attr, labels[:, -1])
                    preds.append(rel_preds)
                    
                    # 8 attributes for 2 objects + 1 relation attribute
                    loss /= 9
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    preds = torch.stack(preds, dim=1)
                    running_loss += loss.item() * inputs.size(0)
                    sum_rows = torch.sum(preds == labels.data, dim=1)
                    running_corrects += torch.sum(sum_rows == 9)
            
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
                    f'/home/nan/PycharmProjects/stylegan2-ada-pytorch/checkpoints/{epoch + 1}.tar')
        
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    datasets = {x: ClevrDataset(split=x) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(dataset=datasets[x], batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
                   for x in ['train', 'val']}
    data_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    # set up model
    model = resnet18(pretrained=False, progress=True)
    num_features = model.fc.in_features
    # [4 shapes + 3 sizes + 9 colors + 3 materials + 3 positions] + 7 relations
    model.fc = nn.Linear(num_features, 51)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, data_sizes, start_epoch=0, num_epochs=50)


if __name__ == '__main__':
    main()