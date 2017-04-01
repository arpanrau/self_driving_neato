

from neural_net.py import NeuralNet
from bag_processor.py import BagProcessor

'''make this file'''

npzfile = np.load('longer-straightest-line.npz')
nn = NeuralNet(learning_rate=.5, images_matrix=npzfile['images_matrix'], input_velocities=npzfile['input_velocities']) #initialize neural net.
nn.optimize_net(iterations=10) #optimize net through 10 iterations.

testfile = np.load('straightest-line.npz')
nn.test_net(test_images=npzfile['images_matrix'], test_velocities=npzfile['input_velocities'])


np.savez('thetas', theta_1=nn.theta_1, theta_2=nn.theta_2)
Seen by Shruti Reyi at 4:58pm
