
Before running the launch file, change the parameters in the bag file to match your file structure. Change the path of the bag file. This script will dump all the jpegs in your home directory under .ros directory.

In order to extract images from a bag file, run
```
roslaunch jpeg.launch
```

After that, get the into a directory with the following commands:
```
cd ~
mkdir test
mv ~/.ros/frame*.jpg test/
```
Change the destination directory as needed
