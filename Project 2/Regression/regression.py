'''
regression.py
-------------
Performs linear regression on given data
------------------------------------------

Author -- Vedant Saboo (CS19B074), K V Vikram (CS19B021)
Team   -- Team 6
Course -- CS 5691 Pattern Recognition and Machine Learning
'''

#!/usr/bin/python

import argparse
import itertools
import threading
import time
import sys

interactive_mode = True
done = True
def animate():
    global done
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     \n')

parser = argparse.ArgumentParser()

# -dtype [1d/2d] -m [ridge/least_squares] -t TRAIN.file -d DEV.file -test TEST.file -i (interactive_mode_flag)
parser.add_argument("-dtype", "--datatype", help="Type of data (1d or 2d)", required=True)
parser.add_argument("-m", "--mode", help="Mode (least_squares or ridge)")
parser.add_argument("-t", "--train", help="Name of file containing training data (relative to current directory)", required=True)
parser.add_argument("-d", "--dev", help="Name of file containing developement data (relative to current directory)", required=True)
parser.add_argument("-test", "--test", help="Name of file containing test data (relative to current directory)", required=True)
parser.add_argument("-i", "--interact", help="Interactive Mode",action='store_true')

args = parser.parse_args()

print(args)

import least_squares_1d
import misc_1d
import misc_2d
import least_squares_2d
import ridge_1d
import ridge_2d

interactive_mode = args.interact

if(args.datatype == '1d'):
    sampling_rate = 20
    max_degree = 7
    
    # global input
    input = lambda *args: misc_1d.input(*args)
    
    if(args.mode == 'least_squares'):
        train = lambda *args: least_squares_1d.train(*args)
        test_model = lambda *args: least_squares_1d.test_model(*args)
    elif(args.mode == 'ridge'):
        train = lambda *args: ridge_1d.train(*args)
        test_model = lambda *args: ridge_1d.test_model(*args)
    else:
        print("Mode not recognized. Using ridge mode as default.")
        train = lambda *args: ridge_1d.train(*args)
        test_model = lambda *args: ridge_1d.test_model(*args)
        
        
elif(args.datatype == '2d'):
    sampling_rate = 200
    max_degree = 6
    
    # global input
    input = lambda *args: misc_2d.input(*args)
    
    if(args.mode == 'least_squares'):
        train = lambda *args: least_squares_2d.train(*args)
        test_model = lambda *args: least_squares_2d.test_model(*args)
    elif(args.mode == 'ridge'):
        train = lambda *args: ridge_2d.train(*args)
        test_model = lambda *args: ridge_2d.test_model(*args)
    else:
        print("Mode not recognized. Using ridge mode as default.")
        train = lambda *args: ridge_2d.train(*args)
        test_model = lambda *args: ridge_2d.test_model(*args)
    
else:
    print("Datatype not recognized: Please put -dtype 1d or -dtype 2d.")
    exit()
    
# input
input_data = input(args.train, args.dev)

if(interactive_mode):
    done = False
    print("Training the model with sample size =", sampling_rate)
    t = threading.Thread(target=animate)
    t.start()
results = train(input_data)
if(interactive_mode):
    done = True
    time.sleep(1)
    
if(interactive_mode):
    done = False
    print("Testing the model...")
    t = threading.Thread(target=animate)
    t.start()
RMS_ERROR = test_model(results, args.test)
if(interactive_mode):
    done = True
    time.sleep(1)
    
print("On Test data, RMS Error was found to be: ", RMS_ERROR)
