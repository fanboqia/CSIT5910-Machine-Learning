# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import transforms

from data import GrayscaleImageFolder
from data.preprocess import tidy_dataset, DATASET_PATH
from data.utils import show_results
from model import ColorizationNet, train, validate


# Check if GPU is available
use_gpu = torch.cuda.is_available()


def setup_datasets():
  # Training dataset
  train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
  train_imagefolder = GrayscaleImageFolder(DATASET_PATH + 'images/train', train_transforms)
  train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=64, shuffle=True)

  # Validation dataset
  val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
  val_imagefolder = GrayscaleImageFolder(DATASET_PATH + 'images/val' , val_transforms)
  val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)

  return train_loader, val_loader


def build_model(pretrained=False):
  model = ColorizationNet()
  # Load model if need
  if pretrained:
    if use_gpu:
        model = torch.load('saved_models/model90.pth')
    else:
        model = torch.load('saved_models/model90.pth', map_location=torch.device('cpu'))
  return model


def run(iter_num=10, pretrained=True, save_model=False, show_result=True):
  tidy_dataset()
  train_loader, val_loader = setup_datasets()
  model = build_model(pretrained=pretrained)
  # Set parameters
  save_images = True
  epochs = iter_num
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
  # Move model and loss function to GPU
  if use_gpu: 
    criterion = criterion.cuda()
    model = model.cuda()
  # Train model
  for epoch in range(epochs):
    # Train for one epoch, then validate
    train(train_loader, model, criterion, optimizer, epoch)
    with torch.no_grad():
      validate(val_loader, model, criterion, save_images, epoch)
  # Save the model
  if save_model:
    torch.save(model, 'saved_models/model.pth')
  # Show the result
  if show_result:
    show_results()


if __name__ == "__main__":
    try:
        run(iter_num=1)
    except:
        run(iter_num=30, pretrained=False, save_model=True)
        print('The model has been saved as ./saved_models/model.pth')
    finally:
        print('The validation result has been saved to ./dataset/outputs')