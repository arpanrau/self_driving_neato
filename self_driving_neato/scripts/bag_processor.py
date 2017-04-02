#!/usr/bin/env python

'''Bag Processor 3000!
	Inputs: gets a bag file
    Outputs: converted images in more pixelated greyscale form as some np array/pickled file.
    Passes images to file to be used by neural_net.py
'''

from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import cv2

class BagProcessor(object):

	def __init__(self):
		self.all_imgs_array = None
		self.all_vel_array = None
		self.latest_vel = np.matrix([0.5,0])

	def img_msg_to_array(self, img_msg):
		'''
		Input: an image name within the directory
		Output: numpy array of the grayscale of that img
		'''
		np_arr = np.fromstring(img_msg.data, np.uint8)
		image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		gray_image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

		reimg = misc.imresize(gray_image_np, 25) # Resizes image to 25% the original
		
		# _, thresh_img = cv2.threshold(reimg,127,255,cv2.THRESH_BINARY)
		# _, thresh_img_2 = cv2.threshold(reimg,220,255,cv2.THRESH_BINARY)
		# _, thresh_img_3 = cv2.threshold(reimg,230,255,cv2.THRESH_BINARY)
		#
		# #plotting multiple things
		# images = [reimg, thresh_img, thresh_img_2, thresh_img_3]
		# titles = ['heat map resized', 'thresh 1', 'thresh 2', 'thresh 3']
		# for i in range(len(images)):
		# 	plt.subplot(2,3,i+1),plt.imshow(images[i])
	    # 	plt.title(titles[i])
		# plt.show()

	 	_, thresh_img = cv2.threshold(reimg,230,255,cv2.THRESH_BINARY)
		return thresh_img

	def flatten_and_add(self, img_array):
		'''
		Input: Receives a numpy array of an img
		Process: Flattens the 2D array to 1D and adds it to master array
		Output: Nothing
		'''
		flat_img = np.asmatrix(img_array).flatten()

		if (self.all_imgs_array!=None):
			self.all_imgs_array = np.concatenate((self.all_imgs_array, flat_img), axis=0)
			self.all_vel_array = np.concatenate((self.all_vel_array, np.array(self.latest_vel)), axis=0)
		else:
			self.all_imgs_array = flat_img
			self.all_vel_array = np.matrix(self.latest_vel)

	def get_imgs(self, bag_file):
		'''
		Input: Bag file with images and input_velocities
		Goes through each msg to pull out images and input_velocities
		And calls function to processes the images
		'''
		bag = rosbag.Bag(bag_file)
		for topic, msg, t in bag.read_messages(topics=['/cmd_vel', '/camera/image_raw/compressed']):
			if (topic=='/cmd_vel'):
				self.latest_vel = np.matrix([msg.linear.x, msg.angular.z])
			if (topic=='/camera/image_raw/compressed'):
				img_array = self.img_msg_to_array(msg)
				self.flatten_and_add(img_array)
		bag.close()

if __name__ == '__main__':
	bp = BagProcessor()
	bag_location = raw_input("Filename for .bag input:\n")
	output_name = raw_input("Filename for .npz output: \n")

	if bag_location=='':
		bag_location='linefollow1.bag'
	if output_name=='':
		output_name='test1'

	bp.get_imgs('../bags/' + bag_location)
	print np.shape(bp.all_imgs_array)
	print np.shape(bp.all_vel_array)
	np.savez(output_name, images_matrix=bp.all_imgs_array, input_velocities=bp.all_vel_array)
