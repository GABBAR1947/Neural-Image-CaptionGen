#sno 0

import os
import time
import tensorflow as tf
import re
from glob import glob
import argparse

import json
from PIL import Image
import pickle
import collections
import random
import skimage

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

import nltk
nltk.download('wordnet')
from nltk import precision
from nltk.translate.bleu_score import sentence_bleu

from scipy import ndimage