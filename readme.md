# CSIT5910 Project - Colori

**Report Title: Colorizing grayscale photographs with convolutional neural network**

**Team Member:  KE Hanlin(20745412), YANG Kuan(20716605),  FAN Boqian(20743139) **

==This is the code repository of the project==

## How to run it?

Before running any codes, please make sure that the dataset has been download. If not, follow these steps:

```reStructuredText
cd path/to/project-colori
cd dataset
!wget http://data.csail.mit.edu/places/places205/testSetPlaces205_resize.tar.gz
!tar -xzf testSetPlaces205_resize.tar.gz
```

If you have problem with wget, just copy the link to your browser, download the dataset to the dataset folder in project-colori and unzip it.

Since the size of the dataset is about 2.2GB, it may takes some time to download.

For training:

```reStructuredText
cd path/to/project-colori
python colorize.py
```

For validation:

Make sure that you satisfied with PyTorch version >= 1.7.0 or you have trained the model before

```reStructuredText
cd path/to/project-colori
python demo.py
```

## How to set up the environment?

This project is based on **Python 3.x** with **PyTorch 1.7.0**. Please download or update the pytorch package with version no lower than 1.7.0, otherwise the pretrained model will not able to use. 

Except PyTorch, you also need to have the following packages:

- torchvision
- numpy
- matplotlib
- scikit-learn

