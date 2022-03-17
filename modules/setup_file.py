# load necessary libraries
import numpy as np
import os
import yaml
from matplotlib import pyplot as plt
from PIL import Image
import re
import datetime as dt
import tensorflow as tf
import pickle

# set up GPUs and output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"