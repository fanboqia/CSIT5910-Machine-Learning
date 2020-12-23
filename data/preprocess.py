import os

DATASET_PATH = 'dataset/'

def tidy_dataset():
  # Move data into training and validation directories
  os.makedirs(DATASET_PATH + 'images/train/class/', exist_ok=True) # 40,000 images
  os.makedirs(DATASET_PATH + 'images/val/class/', exist_ok=True)   #  1,000 images
  for i, file in enumerate(os.listdir(DATASET_PATH + 'testSet_resize')):
    if i < 1000: # first 1000 will be val
      os.rename(DATASET_PATH + 'testSet_resize/' + file, DATASET_PATH + 'images/val/class/' + file)
    else: # others will be train
      os.rename(DATASET_PATH + 'testSet_resize/' + file, DATASET_PATH + 'images/train/class/' + file)
  os.makedirs(DATASET_PATH + 'outputs/color', exist_ok=True)
  os.makedirs(DATASET_PATH + 'outputs/gray', exist_ok=True)
