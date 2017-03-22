#!/usr/bin/env python

'''Neural Net
Does some stuff

Questions we have:
    - what the heck is a bias vector
'''
Class NeuralNet(object):

    self.img_size = 100  #number of pixels in image.
    self.num_images = 5000 #number of images we're feeding into the system
    self.hidden_layer_size = 25  #width of a layer.
    self.output_size = 2     #width of output layer
    images_matrix = zeros(num_images,img_size)
    input_velocities = zeros(num_images, self.output_size)


    #FEED FORWARD NETWORK
    theta_1 = get_rand_theta(self.img_size, self.hidden_layer_size)
    theta_2 = get_rand_theta(self.hidden_layer_size, self.output_size)
    bias_vector = np.ones(self.num_images,1) #num_images x 1

    a_1 = np.concatenate(bias_vector, images_matrix, axis=1) #original image matrix with bias vector added as column. Now num_images x img_size+1 in size.

    z_2 = a_1*np.transpose(theta_1) #unscaled second layer. multiplied a by theta (weights). num_images x hidden_layer_size
    z_2_scaled = sigmoid(z_2) #num_images x hidden_layer_size
    a_2 = np.concatenate(bias_vector, z_2_scaled) #num_images x hidden_layer_size+1

    z_3 = a_2*np.transpose(theta_2) #num_images x output_size
    a_3 = sigmoid(z_3) #num_images x output_size


    #COST FUNCTION
    J = np.sum((input_velocities-a_3)**2)/(2*num_images) #sum all the squared errors, then normalize by number of images (with a 1/2 due to the derivative later)


def get_rand_theta(in_size, out_size):
    random_epsilon = sqrt(6)/(sqrt(in_size+out_size))
    theta = np.random.rand(hidden_layer_size,(img_size+1))*random_epsilon - random_epsilon
    return theta

def sigmoid(matrix): #may not add/divide correctly. check matrix math.
    scaled_matrix = 1/(1+np.exp(-matrix))
        return scaled_matrix
