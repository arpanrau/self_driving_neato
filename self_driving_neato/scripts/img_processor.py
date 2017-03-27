#!/usr/bin/env python

'''Image Processor 3000!
	Inputs: gets absolute path to a dir containing images from the bag file.
    Outputs: converted images in more pixelated greyscale form as some np array/pickled file.
    Passes images to file to be used by neural_net.py
'''

from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

class ImageProcessor(object):

	def __init__(self, dir_path):
		self.dir_path = dir_path
		self.all_imgs_array = None

	def img_to_array(self, img):
		'''
		Input: an image name within the directory
		Output: numpy array of the grayscale of that img
		'''
		img = misc.imread(self.dir_path+img, flatten=True)
		# plt.imshow(img)
		# plt.show()
		reimg = misc.resize(img, 25) # Resizes image to 25% the original
		return reimg

	def flatten_and_add(self, img_array):
		'''
		Input: Receives a numpy array of an img
		Process: Flattens the 2D array to 1D and adds it to master array
		Output: Nothing
		'''
		flat_img = np.asmatrix(img_array).flatten()
		self.all_imgs_array = np.concatenate((self.all_imgs_array, 
										flat_img), axis=0)
