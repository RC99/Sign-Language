import os
import zipfile
local_zip = '/content/Dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp')
zip_ref.close()

from fastai.vision.all import *
from fastai.vision import *
#from fastai.vision.core import *
from fastai.vision.data import *
# Define path to your dataset
path = Path("/tmp/Dataset")

# Define data augmentation and normalization
item_tfms = [Resize(224)]

# Create DataBlock with a train-test split
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   splitter=RandomSplitter(valid_pct=0.2, seed=42),
                   get_y=parent_label,
                   item_tfms=item_tfms)

# Create a DataLoaders object
dls = dblock.dataloaders(path)

# Create a learner with a pre-trained ResNet34 model
learn = cnn_learner(dls, resnet152, metrics=accuracy)

# Fine-tune the model
learn.fine_tune(epochs=5)

learn.export('/tmp/Dataset/exported_model_3.pkl')