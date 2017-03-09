#!/usr/bin/env python
"""General robot controller. Add Keypress functions and key dictionary entries to add functionality"""
import rospy
from sensor_msgs.msg import LaserScan
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

class Control_Robot():

    def __init__(self):
        """ Initialize the robot control, """
        rospy.init_node('robot_controller')
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sleepy = rospy.Rate(2)
        #add subscribers for alternate control modes here

        rospy.on_shutdown(self.stop)
        thread.start_new_thread(self.getKey,())

        # make dictionary that calls functions for teleop
        self.state = {'i':self.forward, ',':self.backward,
                      'l':self.rightTurn, 'j':self.leftTurn,
                      'k':self.stop}
        #modify this if you add any keys for alternative states
        self.acceptablekeys = ['i','l','k',',','j']
        self.linearVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.sendMessage()
        # get key interupt things
        self.settings = termios.tcgetattr(sys.stdin)
        self.key = None
        #current location and orientation
        self.currentx = 0.0
        self.currenty = 0.0
        self.orientation = 0.0
        #proportional controller constants
        self.kturn = .85
        self.kspeed= .1
        #location of person to be followed
        self.personx = 0.0
        self.persony = 0.0
        #location of target for obstacle avoidance
        self.clearx = 0.0
        self.cleary = 0.0

    def getKey(self):
        """ Interupt that gets a non interrupting keypress """
        while not rospy.is_shutdown():
            tty.setraw(sys.stdin.fileno())
            select.select([sys.stdin], [], [], 0)
            self.key = sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            time.sleep(.05)

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
        #self.linearVector  = Vector3(x=0.0, y=0.0, z=0.0)
        #self.angularVector = Vector3(x=0.0, y=0.0, z=-1.0)
        self.linearVector  = Vector3(x=0.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=-1.0)

    def stop(self):
        """ Sets the velocity to stop """
        #print('stop\r')
        self.linearVector  = Vector3(x=0.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.sendMessage()
        #print "currentx = " + str(self.currentx) +'\r'
        #print "currenty = " + str(self.currenty) +'\r'
        #print "orientation = " + str(self.orientation) +'\r'

    def sendMessage(self):
        """ Publishes the Twist containing the linear and angular vector """
        #print('sendMessage\r')
        self.pub.publish(Twist(linear=self.linearVector, angular=self.angularVector))

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
