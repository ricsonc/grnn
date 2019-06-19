import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.misc import imsave, toimage
import scipy.stats as st
from os import makedirs
from os.path import exists, join
import pickle
import constants as const
import os
from PIL import Image, ImageFont, ImageDraw
import random
import matplotlib.pyplot as plt
