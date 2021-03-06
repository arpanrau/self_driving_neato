#!/usr/bin/env python
"""Robot Controller with built - in Self Driving Neural Net. Reads Thetas produced by neural_net.py from file."""
import rospy
from sensor_msgs.msg import CompressedImage
from neato_node.msg import Bump
from geometry_msgs.msg import TwistWithCovariance,Twist,Vector3
import sys
import termios
import math
import time
import tty
import select
import sys
import termios
import cv2
import numpy as np
import cPickle as pickle
import thread
from scipy import misc
from scipy import special



class Control_Robot():

    def __init__(self):
        """ Initialize the robot control, """
        rospy.init_node('robot_controller')
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sleepy = rospy.Rate(2)
        #subscribe to Compressedimage for neural net
        rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.img_msg_to_array)
        #Stop the robot on shutdown
        rospy.on_shutdown(self.stop)
        #Start thread with just getkey
        thread.start_new_thread(self.getKey,())
        # make dictionary that calls functions for teleop
        self.state = {'i':self.forward, ',':self.backward,
                      'l':self.rightTurn, 'j':self.leftTurn,
                      'k':self.stop,'n':self.netdrive}
        #Acceptable keys that won't just stop the robot
        self.acceptablekeys = ['i','l','k',',','j','n']
        #Stops the robot on init just in case 
        self.linearVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.sendMessage()
        # get key interupt things
        self.settings = termios.tcgetattr(sys.stdin)
        self.key = None
        #save last img
        self.last_img = None
        #path to thetas for Learner on Disk
        self.thetapath = 'thetas.npz'
        #load thetas
        self.thetas = np.load(self.thetapath)
        self.theta_1 = self.thetas['theta_1']
        self.theta_2 = self.thetas['theta_2']


    def getKey(self):
        """ Interrupt that gets a non interrupting keypress """
        while not rospy.is_shutdown():
            tty.setraw(sys.stdin.fileno())
            select.select([sys.stdin], [], [], 0)
            self.key = sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            time.sleep(.05)

    ##Keypress control functions

    def forward(self):
        """
            Sets the velocity to forward onkeypress
        """
        #print('forward\r')
        self.linearVector  = Vector3(x=1.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=0.0)

    def backward(self):
        #print('backward\r')
        """ Sets the velocity to backward """
        self.linearVector  = Vector3(x=-1.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=0.0)

    def leftTurn(self):
        #print('leftTurn\r')
        """ Sets the velocity to turn left """
        self.linearVector  = Vector3(x=0.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=1.0)

    def rightTurn(self):
        #print('rightTurn\r')
        """ Sets the velocity to turn right """

        self.linearVector  = Vector3(x=0.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=-1.0)

    def stop(self):
        """ Sets the velocity to stop """
        #print('stop\r')
        self.linearVector  = Vector3(x=0.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.sendMessage()

    def netdrive(self):
        """ Sets the velocity as per neural net """
        #print('netdrive\r')

        #Feedforward from test_net in neural_net.py
        
        test_images = self.last_img
        print str(test_images)+'\r'

        test_bias = np.ones((1,1)) #creating a bias vector for the test set.


        a_1 = np.concatenate((test_bias, test_images), axis=1) #original image matrix with bias vector added as column. Now num_images x img_size+1 in size.

        z_2 = np.dot(a_1, np.transpose(self.theta_1)) #unscaled second layer. multiplied a by theta (weights). num_images x hidden_layer_size
        z_2_scaled = self.sigmoid(z_2) #num_images x hidden_layer_size
        a_2 = np.concatenate((test_bias, z_2_scaled), axis=1) #num_images x hidden_layer_size+1

        z_3 = np.dot(a_2, np.transpose(self.theta_2)) #num_images x output_size
        a_3 = self.sigmoid(z_3) #num_images x output_size

        #Make Vector3 msg to send to robot from

        self.linearVector  = Vector3(x=a_3[0,0], y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=a_3[0,1])
        self.sendMessage()

    def sendMessage(self):
        """ Publishes the Twist containing the linear and angular vector """
        #print('sendMessage\r')
        self.pub.publish(Twist(linear=self.linearVector, angular=self.angularVector))

    ##Helper Functions

    def img_msg_to_array(self, img_msg):
        '''
        Input: img message type
        Output: flattened numpy array of the grayscale of that img
        '''
        np_arr = np.fromstring(img_msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray_image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        reimg = misc.imresize(gray_image_np, 25) # Resizes image to 25% the original
        _, thresh_img = cv2.threshold(reimg,230,255,cv2.THRESH_BINARY)
        flat_img = np.asmatrix(thresh_img).flatten()

        self.last_img = flat_img


    def sigmoid(self, matrix): 
        """Helper function that performs sigmoid on a matrix"""
        scaled_matrix = special.expit(matrix)
        return scaled_matrix

    ##Main

    def run(self):

        while self.key != '\x03' and not rospy.is_shutdown():
            if self.key in self.acceptablekeys:
                #if an acceptable keypress, do the action
                self.state[self.key].__call__()
            else:
                # on any other keypress, stop the robot
                self.state['k'].__call__()
            self.sendMessage()
        self.sleepy.sleep()

control = Control_Robot()
control.run()
