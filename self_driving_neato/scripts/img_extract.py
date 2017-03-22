#!/usr/bin/env python

'''For image extraction and processing from the neato.

    Inputs: subscribe to image topic (/camera_raw).
    Outputs: converted images in more pixelated greyscale form as some np array/pickled file.
    Passes images to file to be used by neural_net.py

    Current questions:
        - what output filetype are we using for our NN? (wait to decide until have NN)
'''
