#!/usr/bin/env python

'''Neural Net
Does some stuff

Questions we have:
    - does this run once and does it most optimally for this dataset based on one run?
'''

import numpy as np
import math

class NeuralNet(object):

    def __init__(self):
        self.img_size = 100  #number of pixels in image.
        self.num_images = 5000 #number of images we're feeding into the system
        self.hidden_layer_size = 25  #width of a layer.
        self.output_size = 2     #width of output layer
        self.images_matrix = np.zeros((self.num_images,self.img_size))
        self.input_velocities = np.zeros((self.num_images, self.output_size))
        
    def init_with_bag(self, images_matrix, input_velocities):
        pass
        self.images_matrix = images_matrix
        self.input_velocities = input_velocities
        (self.num_images, self.img_size) = np.shape(images_matrix)

    def feed_forward_and_back_prop(self):
        #FEED FORWARD NETWORK
        theta_1 = self.get_rand_theta(self.img_size, self.hidden_layer_size)
        theta_2 = self.get_rand_theta(self.hidden_layer_size, self.output_size)
        bias_vector = np.ones((self.num_images,1)) #num_images x 1

        a_1 = np.concatenate((bias_vector, self.images_matrix), axis=1) #original image matrix with bias vector added as column. Now num_images x img_size+1 in size.

        z_2 = np.dot(a_1, np.transpose(theta_1)) #unscaled second layer. multiplied a by theta (weights). num_images x hidden_layer_size
        z_2_scaled = self.sigmoid(z_2) #num_images x hidden_layer_size
        a_2 = np.concatenate((bias_vector, z_2_scaled), axis=1) #num_images x hidden_layer_size+1

        z_3 = np.dot(a_2, np.transpose(theta_2)) #num_images x output_size
        a_3 = self.sigmoid(z_3) #num_images x output_size

        #COST FUNCTION
        cost = np.sum((self.input_velocities-a_3)**2)/(2*self.num_images) #sum all the squared errors, then normalize by number of images (with a 1/2 due to the derivative later)
    
        #BACK PROPAGATION (INCOMPLETE)
        delta_3 = np.subtract(a_3, self.input_velocities) #difference between predicted and actual outputs
        a = np.dot(delta_3, theta_2)
        b = np.concatenate((bias_vector, self.sigmoidGradient(z_2)), axis=1)

        delta_2 = np.multiply(a, b)[:,1:]  

        #these don't make as much sense. come back and interpret better.
        big_delta_1 = np.dot(np.transpose(delta_2), a_1)
        big_delta_2 = np.dot(np.transpose(delta_3), a_2)

        theta_1_grad = big_delta_1/self.num_images
        theta_2_grad = big_delta_2/self.num_images

        return cost, theta_1_grad, theta_2_grad

    def get_rand_theta(self, in_size, out_size):
        random_epsilon = math.sqrt(6)/(math.sqrt(in_size+out_size))
        theta = np.random.rand(out_size,(in_size+1))*2*random_epsilon - random_epsilon
        return theta

    def sigmoid(self, matrix): #may not add/divide correctly. check matrix math.
        scaled_matrix = 1/(1+np.exp(-matrix))
        return scaled_matrix

    def sigmoidGradient(self, matrix):
        g = np.zeros(np.shape(matrix))  
        #inverse of sigmoid is sigmoid*(1-sigmoid)
        g_z = self.sigmoid(matrix)
        g = g_z*(1-g_z)
        return g

if __name__ == '__main__':
    npzfile = np.load('longer-straightest-line.npz')
    nn = NeuralNet()
    nn.init_with_bag(npzfile['images_matrix'], npzfile['input_velocities'])
    cost, theta_1_grad, theta_2_grad = nn.feed_forward_and_back_prop()
    print cost
    print np.shape(theta_1_grad), np.shape(theta_2_grad)