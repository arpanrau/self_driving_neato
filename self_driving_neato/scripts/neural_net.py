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

class NeuralNet(object):

    def __init__(self, images_matrix, input_velocities, learning_rate=.5):
        #Set training values and sizes based on inputs:
        self.images_matrix = images_matrix
        self.input_velocities = input_velocities
        (self.num_images, self.img_size) = np.shape(images_matrix)
        self.learning_rate = learning_rate

        #Initialized once at creation of net:
        self.hidden_layer_size = 4  #width of a layer.
        self.output_size = 2     #width of output layer
        self.bias_vector = np.ones((self.num_images,1))

        #Updated every iteration:
        self.theta_1 = self.get_rand_theta(self.img_size, self.hidden_layer_size) #initialize to random thetas to start.
        self.theta_2 = self.get_rand_theta(self.hidden_layer_size, self.output_size) #initialize to random thetas to start.
        self.cost = 0.0
        self.theta_1_grad = 0.0
        self.theta_2_grad = 0.0

    def optimize_net(self, iterations=100):
        start_time = time() #for timing purposes. Prints at end before showing the plot.

        cost_list = np.zeros((iterations,1))

        for i in range(iterations):
            print "iteration: ",i, "\t \t cost: ",self.cost
            _ = self.feed_forward_and_back_prop()
            cost_list[i] = self.cost

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

    def test_net(self, test_images=[], test_velocities=[]):
        '''Test the net after optimizing. This function will take the optimized thetas
        and a test set, then output accuracy of predicted velocities against test set velocities.'''
        #FEED FORWARD
        (num_test_images, num_pixels) = test_images.shape #number of rows in test set is number of images.
        test_bias = np.ones((num_test_images,1)) #creating a bias vector for the test set.
        a_1 = np.concatenate((test_bias, test_images), axis=1) #original image matrix with bias vector added as column. Now num_images x img_size+1 in size.

        z_2 = np.dot(a_1, np.transpose(self.theta_1)) #unscaled second layer. multiplied a by theta (weights). num_images x hidden_layer_size
        z_2_scaled = self.sigmoid(z_2) #num_images x hidden_layer_size
        a_2 = np.concatenate((test_bias, z_2_scaled), axis=1) #num_images x hidden_layer_size+1

        z_3 = np.dot(a_2, np.transpose(self.theta_2)) #num_images x output_size
        a_3 = self.sigmoid(z_3) #num_images x output_size

        #ACCURACY FUNCTIONS
        linear_accuracy = np.sum((test_velocities[:][0]-a_3[:][0])**2)/(2*num_test_images)
        angular_accuracy = np.sum((test_velocities[:][1]-a_3[:][1])**2)/(2*num_test_images)
        print 'Linear Accuracy: ', (1-linear_accuracy)*100, '%'
        print 'Angular Accuracy: ', (1-angular_accuracy)*100, '%'
        print 'Mean Accuracy', (1-(linear_accuracy+angular_accuracy)/2)*100, '%'

        mse_accuracy = np.sum((test_velocities-a_3)**2)/(2*num_test_images) #sum all the squared errors, then normalize by number of images (with a 1/2 to cancel out the derivative/sigmoid that's taken later)
        #this may be wrong.
        print 'Probably Incorrect \"Accuracy\": ', (1-mse_accuracy)*100, '%'



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

        big_delta_1 = np.dot(np.transpose(delta_2), a_1)
        big_delta_2 = np.dot(np.transpose(delta_3), a_2)

        #calculate gradients of thetas
        self.theta_1_grad = big_delta_1/self.num_images
        self.theta_2_grad = big_delta_2/self.num_images

        #calculate new theta vectors to use in next iteration.
        self.theta_1 -= self.learning_rate*self.theta_1_grad
        self.theta_2 -= self.learning_rate*self.theta_2_grad

        return self.cost

    def get_rand_theta(self, in_size, out_size):
        random_epsilon = sqrt(6)/(sqrt(in_size+out_size))
        theta = np.random.rand(out_size,(in_size+1))*2*random_epsilon - random_epsilon
        return theta

    def sigmoid(self, matrix):
        #handmade sigmoid function, got overflow errors often so we switched to the scipy function.
        #scaled_matrix = 1/(1+np.exp(-matrix))
        scaled_matrix = special.expit(matrix) #scipy.special matrix exponentiation to avoid overflow errors
        return scaled_matrix

    def sigmoidGradient(self, matrix):
        sg = np.zeros(np.shape(matrix))
        #inverse of sigmoid is sigmoid*(1-sigmoid)
        sigmoid_matrix = self.sigmoid(matrix)
        sg = sigmoid_matrix*(1-sigmoid_matrix)
        return sg

if __name__ == '__main__':
    #Set inputs to learn and test on, then learning rate and number of epochs
    inputfilename = raw_input("Path for NPZ file to Learn on :\n")
    testfilename = raw_input("Path for NPZ file to Test on :\n")
    learning_rate= raw_input("Learning Rate :\n")
    iterations = raw_input("Iterations :\n")

    #Set default values, for if the user doesn't want to type filenames out every time and instead just hits enter.
    if inputfilename=='':
        inputfilename = 'linefollow1.npz'
    if testfilename=='':
        testfilename = 'linefollow2.npz'
    if learning_rate=='':
        learning_rate='.9'
    if iterations=='':
        iterations='10'

    learning_rate=float(learning_rate)
    iterations = int(iterations)
    npzfile = np.load(inputfilename)
    testfile = np.load(testfilename)

    #initialize neural net.
    nn = NeuralNet(learning_rate=learning_rate, images_matrix=npzfile['images_matrix'], input_velocities=npzfile['input_velocities'])
    #train net with given amount of epochs (iterations)
    nn.optimize_net(iterations=iterations)
    #validate net with test set
    nn.test_net(test_images=npzfile['images_matrix'], test_velocities=npzfile['input_velocities'])

    #save optimized theta values into a pickled numpy array. Will be used in robot_controller.py
    np.savez('thetas', theta_1=nn.theta_1, theta_2=nn.theta_2)
