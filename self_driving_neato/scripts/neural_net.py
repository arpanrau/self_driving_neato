#!/usr/bin/env python

'''Neural Net
Initializes a neural net and optimizes it over some number of iterations defined by the user.
Implemented fully from scratch.

Inputs: Load npz file from bag_processor.
Outputs: graph of cost over iterations, theoretically also accuracy validation.
    (Still needs integration with real-time output velocities if we want to drive a neato from this).
'''

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from scipy import special
from load_mnist import mnist
from sklearn.metrics import accuracy_score

class NeuralNet(object):

    def __init__(self, images_matrix, input_velocities, learning_rate=.5):
        self.images_matrix = images_matrix
        self.input_velocities = input_velocities
        (self.num_images, self.img_size) = np.shape(images_matrix)
        self.learning_rate = learning_rate

        # Initialized once at creation of net:
        self.hidden_layer_size = 256  #width of a layer.
        self.output_size = 10     #width of output layer
        
        self.bias_vector = np.ones((self.num_images,1))

        #Updated every iteration:
        self.theta_1 = self.get_rand_theta(self.img_size, self.hidden_layer_size) #initialize to random thetas to start.
        self.theta_2 = self.get_rand_theta(self.hidden_layer_size, self.output_size) #initialize to random thetas to start.
        self.cost = 0.0
        self.accuracy = 0.0
        self.theta_1_grad = 0.0
        self.theta_2_grad = 0.0

    def optimize_net(self, iterations=100):
        start_time = time() #for timing purposes. Prints at end before showing the plot.

        cost_list = np.zeros((iterations,1))
        accuracy_list = np.zeros((iterations,1))

        for i in range(iterations):
            self.cost, self.accuracy = self.feed_forward_and_back_prop()
            print "iteration: ",i, "\t \t cost: ",
            cost_list[i] = self.cost
            accuracy_list[i] = self.accuracy

        #Time the optimization portion. since it does this in real time, this will include other things making your computer slow.
        print "--- NN Optimization time: %s seconds ---" % (time() - start_time)
        print "Close figure to end program."

        #Plot dat data, yo
        fig2 = plt.figure(2)
        plt.plot(range(iterations),cost_list, 'b*-')
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.title('Cost over iterations. Learning rate: '+str(self.learning_rate)) #may not work with adding variable
        plt.show()

        #Plot another data, yo
        fig3 = plt.figure(3)
        plt.plot(range(iterations),accuracy_list, 'b*-')
        plt.grid(True)
        plt.xlabel('iteration')
        plt.ylabel('accuracy')
        plt.title('Accuracy over iterations. Learning rate: '+str(self.learning_rate)) #may not work with adding variable
        plt.show()

    def test_net(self, test_images=[], test_velocities=[]):
        '''Test the net after optimizing. This function will take the optimized thetas
        and a test set, then output accuracy of predicted velocities against test set velocities.'''
        #FEED FORWARD

        (num_test_images, num_pixels) = test_images.shape #number of rows in test set is number of images.
        test_bias = np.ones((num_test_images,1)) #creating a bias vector for the test set.
        a_1 = np.concatenate((test_bias, test_images), axis=1) #original image matrix with bias vector added as column. Now num_images x img_size+1 in size.
        print test_bias.shape, test_images.shape, test_velocities.shape

        # Randomly generated
        # test_bias = np.ones((num_test_images,1)) #creating a bias vector for the test set.
        # test_images = self.get_rand_theta(num_pixels-1,num_test_images) #initialize to random thetas to start.
        # test_velocities = self.get_rand_theta(2-1,num_test_images)
        # print test_bias.shape, test_images.shape, test_velocities.shape
        # a_1 = np.concatenate((test_bias, test_images), axis=1) #original image matrix with bias vector added as column. Now num_images x img_size+1 in size.

        z_2 = np.dot(a_1, np.transpose(self.theta_1)) #unscaled second layer. multiplied a by theta (weights). num_images x hidden_layer_size
        z_2_scaled = self.sigmoid(z_2) #num_images x hidden_layer_size
        a_2 = np.concatenate((test_bias, z_2_scaled), axis=1) #num_images x hidden_layer_size+1

        z_3 = np.dot(a_2, np.transpose(self.theta_2)) #num_images x output_size
        a_3 = self.sigmoid(z_3) #num_images x output_size
        print a_3
        print a_3.shape

        input_digits = np.argmax(a_3, axis=1)
        actual_digits = np.argmax(test_velocities, axis=1)

        print "TEST ACCURACY", accuracy_score(input_digits, actual_digits)

    def feed_forward_and_back_prop(self):
        '''With one call of this function, goes through an iteration of
        feedforward calculation of theoretical velocity outputs, calculates costs,
        then backpropagates to reach better values for theta.
        Gets called many times in optimize_net().
        '''
        #FEED FORWARD NETWORK
        a_1 = np.concatenate((self.bias_vector, self.images_matrix), axis=1) #original image matrix with bias vector added as column. Now num_images x img_size+1 in size.

        z_2 = np.dot(a_1, np.transpose(self.theta_1)) #unscaled second layer. multiplied a by theta (weights). num_images x hidden_layer_size
        z_2_scaled = self.sigmoid(z_2) #num_images x hidden_layer_size
        a_2 = np.concatenate((self.bias_vector, z_2_scaled), axis=1) #num_images x hidden_layer_size+1

        z_3 = np.dot(a_2, np.transpose(self.theta_2)) #num_images x output_size
        a_3 = self.sigmoid(z_3) #num_images x output_size

        #COST FUNCTION
        self.cost = np.sum((self.input_velocities-a_3)**2)/(2*self.num_images) #sum all the squared errors, then normalize by number of images (with a 1/2 to cancel out the derivative/sigmoid that's taken later)

        #BACK PROPAGATION
        delta_3 = np.subtract(a_3, self.input_velocities) #difference between predicted and actual outputs
        a = np.dot(delta_3, self.theta_2)
        b = np.concatenate((self.bias_vector, self.sigmoidGradient(z_2)), axis=1)

        delta_2 = np.multiply(a, b)[:,1:] #element-wise multiplication, then slicing of the bias vector.

        #TODO: make sure we all understand this operation.
        big_delta_1 = np.dot(np.transpose(delta_2), a_1)
        big_delta_2 = np.dot(np.transpose(delta_3), a_2)

        #calculate gradients of thetas
        self.theta_1_grad = big_delta_1/self.num_images
        self.theta_2_grad = big_delta_2/self.num_images

        #calculate new theta vectors to use in next iteration.
        self.theta_1 -= self.learning_rate*self.theta_1_grad
        self.theta_2 -= self.learning_rate*self.theta_2_grad
        
        input_digits = np.argmax(a_3, axis=1)
        actual_digits = np.argmax(self.input_velocities, axis=1)

        print "TRAIN ACCURACY", accuracy_score(input_digits, actual_digits)

        return self.cost, accuracy_score(input_digits, actual_digits)

    def get_rand_theta(self, in_size, out_size):
        """
        Initialize thetas according the size of adjacent layers
        """
        random_epsilon = sqrt(6)/(sqrt(in_size+out_size)) # Initialize it to very small non-zero numbers
        # With the range (-epsilon, epsilon)
        theta = np.random.rand(out_size,(in_size+1))*2*random_epsilon - random_epsilon
        return theta

    def sigmoid(self, matrix):
        """
        Sigmoid activation fuction applied to each element of the matrix
        """ 
        scaled_matrix = special.expit(matrix)
        return scaled_matrix

    def sigmoidGradient(self, matrix):
        """
        Inverse of sigmoid is sigmoid*(1-sigmoid) which is used to calculate gradient
        """
        g = np.zeros(np.shape(matrix))
        g_z = self.sigmoid(matrix)
        g = g_z*(1-g_z)
        return g

if __name__ == '__main__':
    # Load the MNIST data
    trX,teX,trY,teY = mnist()

    # Making sure the shapes are what we expect them to be
    print trX.shape, teX.shape, trY.shape, teY.shape

    learning_rate=0.01
    iterations = 100
    
    nn = NeuralNet(learning_rate=learning_rate, images_matrix=trX, input_velocities=trY) # Initialize neural net.
    nn.optimize_net(iterations=iterations) # Optimize net through x iterations.
    
    nn.test_net(test_images=teX, test_velocities=teY) # Test the neural net's accuracy
