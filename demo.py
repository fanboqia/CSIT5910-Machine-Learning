import torch
import torch.nn as nn
from torchvision import transforms

from data import GrayscaleImageFolder
from data.preprocess import tidy_dataset, DATASET_PATH
from data.utils import show_results
from model import ColorizationNet, validate


# preprocessing
tidy_dataset()

# Check if GPU is available
use_gpu = torch.cuda.is_available()


# Validation dataset
val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
val_imagefolder = GrayscaleImageFolder(DATASET_PATH + 'images/val' , val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)


model = ColorizationNet()
try:
    if use_gpu:
        model = torch.load('saved_models/model100.pth')
    else:
        model = torch.load('saved_models/model100.pth', map_location=torch.device('cpu'))
except:
    if use_gpu:
        model = torch.load('saved_models/model.pth')
    else:
        model = torch.load('saved_models/model.pth', map_location=torch.device('cpu'))
finally:
    # Set parameters
    save_images = True
    # Move model and loss function to GPU
    if use_gpu: 
        criterion = criterion.cuda()
        model = model.cuda()
    # Validate model
    with torch.no_grad():
        validate(val_loader, model, criterion, save_images, 0)

    # Show the validation results
    show_results()
