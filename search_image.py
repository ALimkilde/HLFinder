# This module implements a search image data structure

import numpy as np
from math import floor
from scipy.ndimage import maximum_filter, minimum_filter
import matplotlib.pyplot as plt

class SearchImage:
   """
   Represent an image search data structure with search methods
   """

   def __init__(self, image, original_px_size_m, block_size):
       self.image = image
       self.original_px_size_m = original_px_size_m
       self.block_size = block_size
       self.n, self.m = image.shape

   @classmethod
   def coarse(cls, image, px_size_m, coarse_px_size_m, filter_type):
       """
       Efficient coarse representation of an image using SciPy max/min filtering.
       """

       # Compute block size (how many original pixels → one coarse pixel)
       scale =  float(coarse_px_size_m) / float(px_size_m)
       print(f"scale: {scale}")
       block = max(1, floor(scale))
       print(f"block: {block}")

       if block <= 1:
           return image.copy()

       # Pick SciPy filter
       if filter_type == 'max':
           f = maximum_filter
       elif filter_type == 'min':
           f = minimum_filter
       else:
           raise ValueError("filter_type must be 'max' or 'min'")

       # Apply filter with a window equal to the coarse block size
       # mode='nearest' avoids edge artifacts
       filtered = f(image, size=(block, block), mode='nearest')

       # Downsample by taking every block-th pixel
       if image.ndim == 2:
           coarse = filtered[::block, ::block]
       else:
           coarse = filtered[::block, ::block, :]

       return cls(coarse, px_size_m, block)

   def plot(self):
       plt.figure()
       plt.imshow(self.image)

   def get_original_px(self, x, y):

       xorig = x * self.block_size
       yorig = y * self.block_size

       return xorig, yorig
       
   def get_original_range(self, xmin, xmax, ymin, ymax):

       xmin_orig = get_original_px(xmin)
       xmax_orig = get_original_px(xmax)
       ymin_orig = get_original_px(ymin)
       ymax_orig = get_original_px(ymax)

       return xmin_orig, xmax_orig, ymin_orig, ymax_orig



        

