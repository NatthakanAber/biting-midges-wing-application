'''
to add new pretrained model, see inspect_pretrained_model.py
'''

import torch
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import ImageFromFileDataset

def data_setup(train_set, test_set, y_train, y_test, class_names, dataset_path, pretrained_transforms, batch_size):
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = ImageFromFileDataset.ImageFromFileDataset(file_df=train_set, labels=y_train, class_names=class_names, root_dir=dataset_path,  transform=pretrained_transforms)
    val_dataset = ImageFromFileDataset.ImageFromFileDataset(file_df=test_set, labels=y_test, class_names=class_names, root_dir=dataset_path, transform=pretrained_transforms)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=1, shuffle=False)
    }
    dataset_sizes = {
        'train': len(train_set),
        'val': len(test_set)
    }

    return dataloaders, dataset_sizes

def model_setup(model_name, no_of_classes, frozen_base):
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(device)

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if frozen_base:
        # Set up the model with pretrained weights and freeze base layers
        model, weights = eval(model_name + '(no_of_classes, frozen_base=True)')
    else:
        # Set up the model with pretrained weights and not freezing base layers
        model, weights = eval(model_name + '(no_of_classes, frozen_base=False)')

    # seed it to the target device
    model = model.to(device)

    # Get the transforms used to create our pretrained weights
    pretrained_transforms = weights.transforms()

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters()) #, default lr=0.001

    return model, pretrained_transforms, criterion, optimizer, device

def EfficientNet_B0(no_of_classes, frozen_base):
    # Set up the model with pretrained weights
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights) #.to(device)

    if frozen_base:
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        print('Not frozen_base')

    # Recreate the classifier layer
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=no_of_classes,
                        bias=True))
    return model, weights