# Image Classification with PyTorch
A CNN model for solving a binary classification problem to detect whether a person wearing a mask properly, with [PyTorch](https://pytorch.org/).
![alt text](https://github.com/uriMen/image-classification-with-pytorch/tree/main/img/sample_imgs.png?raw=true)

Model architecture is based on VGG-11 with some modifications (dropouts, batchNorm, etc.)

#### Included in the train.py file
- DataSet class, which includes pre-processing methods (resizing, normalizing) and data augmentation.
- Model class
- Training process
