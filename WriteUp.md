# Self-Driving Neato

Lauren Gulland, Shruti Iyer, Arpan Rau

*Computer Vision Project for [Computational Robotics](https://sites.google.com/site/comprobo17) as taught by [Paul Ruvolo](https://github.com/paulruvolo/comprobo17) at Olin College of Engineering, Spring 2017*


## Goal
The major goal of the project was to use computer vision and machine learning to train our Neato to drive autonomously around an arbitrary tape-marked course. By setting up a neural network to take processed camera data from the neato while under teleoperated human control, we wanted to be able to predict robot wheel velocities based on input camera images when not driven by a human.

The learning goal of the project was to implement as much as we can by ourselves to try to gain a deep understanding of what actually makes neural nets work. We were far more concerned in this project with implementing a pipeline that we could theoretically use to train our robot to drive than actually training our robot to drive; in other words, we were willing to trade good results for deeper learning.


## How the system works

### Training the Neural Net

To train the neural net, we use pre-recorded bags of data where a human drives the neato around a path marked with tape. Once we read the images from the bag, we use binary thresholding to convert an RGB image into B/W image, which makes the white tape pop out on a black background. We also resize the image to 10% the original size to reduce the size of the data. The post-processed pixel values get flattened and added as a row to a numpy matrix and the linear and angular velocities corresponding each image get added to another numpy matrix. We use the matrix with image data as an input to our neural net to output velocities. With the output error calculated using  the velocities matrix, we correct the weights of the neural net. After training, we save these weights for driving around the neato.
### Driving Neato using pre-trained Neural Net

In order to control the Neato, the robot controller reads pickled theta values from our learner. It then pulls live images from the camera, flattening and thresholding them in the same way that our bag processor does to the training data. It then uses the pre-trained feedforward neural net to compute which messages to send to cmd_vel.  We intentionally built our robot controller to run the neural net controller with teleoperated keyboard interrupts so we could avoid running into walls and do little things like ‘stop and turn 20 degrees’ while tweaking and testing the neural net controller. 


## Design Decisions

We made two large decisions that set the scope of our project. First, we decided to use a neural net as a linefollower. While we understand that there are much better ways to make linefollowers, our primary motivation was learning more about machine learning.

Second, we decided to do as much implementation as we could by hand instead of using someone else’s implementation. This handicapped us in terms of both bandwidth and ability - we spent a lot of time learning things that we could have implemented with a function call to something somebody else had built - but enabled us to fulfill our learning goals of building deep understanding. 

Two more specific design decisions we made were in the functionality of our neural net: implementation and choice of cost function, and topology of the net itself. Because our neural net was based off of linear regression, we were choosing between mean-squared error and mean-absolute-percentage error, which are both commonly used for this category of problem. We chose mean-squared error because mean-absolute-percentage error, although it looked promising, did not deal with zeros in the validation data well. The other specific decision we made was in topology of our neural net. We started simple, with one hidden layer, which served us well in the beginning but started to lose accuracy on more complicated training sets. When we ran into this problem, we experimented with adding a hidden layer and manipulating the general shapes, but were surprised when our accuracy didn’t increase in any significant manner, so we returned to the single hidden layer and improved our net in other ways.	


## Code Architecture

![Code Arch Diagram](https://github.com/arpanrau/self_driving_neato/blob/master/code_structure_diagram_lowres.png)

### `Bag_Processor.py`
Using a python library called ‘rosbag’, we extracted compressed images and velocities. In order to process the image, we first decompress the images. We then convert those images into B/W  with binary thresholding so that we see the line of tape as a thick white line in a black background. The B/W images are then resized to have 10% of the original width and height (64x48 pixels) so that we don’t overwhelm the neural net with too many features. For ease of data usage and matrix math, the resized image pixels are flattened and added as a row to a matrix and the velocities are added to a different matrix.

### `Neural_Net.py`

Neural_net.py creates, optimizes, and validates a neural net given processed training/testing data. To instantiate the neural net, one creates a NeuralNet object, which takes in a pickled file from bag_processor containing training data with an image matrix and a matrix of associated velocities (see above bag_processor.py section). On initialization, the net creates a randomly generated theta matrix, and sets cost and other values to be calculated to zero. After this, the user then calls the optimize_net() function, which runs a feedforward neural net with one hidden layer, then calculates cost for the given theta values via a Mean-Squared Error function. From this cost, it back propagates to improve and update the theta values. This cycle of feedforward and backpropagation continues for however many epochs the user decides, and cost should decrease on each epoch. Once the net is optimized for a low cost, the theta values are saved into a pickled numpy matrix to be used in robot controller. 

After optimizing the net, the user will then call test_net(), which takes in a different pickled training set and runs the same feedforward net with the optimized thetas on the input test images from this pickled dataset. Once the predicted velocities are calculated, we run an accuracy function, which is essentially a modified version of our MSE cost function, to validate the predicted velocities against those which came in the pickled training set. 

The three outputs of the Neural_Net are: a pickled numpy array of optimized theta values (to be used in robot_controller.py), a graph of cost over number of epochs (for user visualization), and a final accuracy from the test set (also for validation and user visualization). 
	
### `Robot_Controller.py`

`robot_controller.py` is the piece of software that actually drives the robot. As such, it has two primary responsibilities: reading both the thetas from file and the live images and feeding them into a neural net to drive the robot, and allowing for teleoperated control should a human operator wish to intervene. 

The script is built such that it runs two threads simultaneously.  One thread is the normal ROS thread, and the other runs a function called `getKey()` that reads the keyboard values. When a keyboard value is read, it is mapped to a function that puts the main thread in a different state via a dictionary.  There is a state for each teleop operation (forward, backward, etc), and a state for neural net control. 

In order to utilize neural net control, Robot Controller relies on the helper function img_message_to_array from `bag_processor.py`, which makes a flattened numpy array out of a ROS compressed image message. When a new image message is recieved, a variable called `last_img` which contains the last recieved image is updated. This structure allows the neural net to run at whatever pace it can and ensures it grabs the most recent avaliable image when it starts. 

The neural net itself reads theta values from a pickle file and performs the exact same feedforward operation as `neural_net.py` does on last_img to produce and send a twist message to /cmd_vel. 

## Challenges Faced

When we first trained our neural nets on simple paths of mostly straight lines, we got a very high accuracy of ~99%. But the neural net didn’t perform well on curved paths. 

In order to see if the problem was in our net or in the data we were training on, we took a brief pause to validate our neural net using the MNIST dataset. The code can be found in our mnist branch here. The accuracy of the neural net increases with more training and more iterations. After 100 iterations, we achieved an accuracy of ~42% for both train and test data. This is better than random guessing accuracy of 10%. Even though this accuracy isn’t that high, we can at least see that the neural network works as expected. 


![Accuracy of the MNIST data with the neural net](https://github.com/arpanrau/self_driving_neato/blob/master/mnist_accuracy_function.png)


## Future Improvements

### Use the Neural Net to predict only Angular velocity

One thing that was evident was that there was significant amount of forward speed variance in our training data. Because the forward speed mapped more to my comfort in driving and less to the part of the course that the robot was navigating, we believe that we could have obtained better results had we set a constant forward speed and trained both myself and the robot to drive the course at that speed. In this way, we could go from one output variable to two while also hopefully reducing the variance in our data set that isn’t correlated to our video input. 

### Using More Normalized Data

![Our Training Path](https://github.com/arpanrau/self_driving_neato/blob/master/robot_course.jpg)

As the path we trained the neural net on was mostly straight lines (about 70% by length), a net that was trained to drive straight forwards would be right about 70% of the time. We suspect that this was a primary reason why our final product failed to successfully follow the line well (it just drove straight). In the future, we think we would use a much curvier line to force the neural net to learn to respond to the camera inputs in order to be correct (as opposed to just going arbitrarily forward).

## Lessons Learned/Insights Gained

__Arpan__: I personally really enjoyed being able to open the black box of math that neural networks have been to me in the past. Specific to machine learning, I learned that choosing the correct cost function is very important to making a neural network work, as is picking the correct training data. The revelation that we had been teaching our robot to drive on a mostly straight line and that it therefore was learning that it could go straight most of the time and still work was an important one to me.  

__Lauren__: I learned a ton about the low-level details of neural networks and machine learning, which is really exciting and definitely met my individual learning goals for this project. I think our balance of pair programming and stepping through the basic structure of our feedforward and backprop functions all together with being able to parallelize on less concept-heavy topics really benefitted us, and is something to be mindful of on future projects. On future projects, I want to make sure not to underestimate the importance of validation, but on this project, since our learning goals weren’t to make something that worked spectacularly, it honestly didn’t matter that much, so this was a good project to learn that on. In the future, I want to try a higher-level approach (e.g. keras, sci-kit learn) to machine learning to see how different it is, and then hopefully dive down into more nitty-gritty (e.g Tensor Flow, individual implementation) when it’s necessary to improve the nets. 

__Shruti__: My biggest learning goal for this project was to understand the mystery behind neural networks better. We implemented the whole feedforward and backpropagation and researched why things were a certain way. For a short while, we considered using Tensorflow for the neural net regression. So I ended up learning about Tensorflow and writing short programs which was super cool. The biggest lesson learned was we can make or break the neural net according to the data we give it. If we had given our neural net a variety of images and corresponding velocities, it might have learned to drive in paths other than straight lines. We also should have validated the neural net architecture with MNIST very early on in the project to see where in the pipeline we are losing accuracy.

