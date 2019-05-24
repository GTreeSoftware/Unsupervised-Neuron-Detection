import numpy as np
from skimage import morphology
import tifffile
from config import opt
import skimage.util
import time
from dataset.generatePatches import *

def denoise_by_connection(image, min_size):
    image = image > 0.8
    image = image.astype(np.bool)
    image_new = morphology.remove_small_objects(image, min_size, connectivity=2)
    image_new = image_new.astype(np.int)
    image_new = image_new > 0.8
    image_new = image_new.astype(np.int)
    return image_new

def skeleton_by_morphology(image):
    image = skimage.util.invert(image)
    image_skeleton = morphology.skeletonize_3d(image)
    return image_skeleton

def denoise_by_erosion(image):
    image_erosion = morphology.erosion(image)
    return image_erosion


