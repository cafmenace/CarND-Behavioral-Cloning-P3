#Import pandas library to read data from csv file.
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import cv2
import os, sys

#Directory for training data
dir = "./data/data"
imagedir = os.path.join(dir, "IMG")

#list = os.listdir(imagedir) 
#number_files = len(list)
#print("Number of still frames: {0}".format(number_files))

filename=os.path.join(dir, "driving_log.csv")
df = pd.read_csv(filename, header= 0, skipinitialspace=True)
#print("Total number of rows in table before filtering is: {0} ".format(len(df)))

#Dropping some of the steering data within range: -0.1 < steering < 0.1 
df = df.drop(df[:6500][(df.steering > -0.1)&(df.steering < 0.1)].index)

#Split data into training and validation set
train_data, valid_data = train_test_split(df[['center', 'left','right','steering']], test_size=0.2)


#==================================================================================================
#                             Skew_Image FUNCTION: Augments training data
#==================================================================================================
def Skew_Image(img, img_type, steering):
    import random
    import numpy as np
    
    '''
    Randomly performs rotation on images and adds an a steering offset based on 
    image rotation angle. Some of the images are also randomly selected to alter 
    the brightness.
    '''
      
    rows, cols, ch = img.shape
    
    #Declare min and max steering angles
    left_turn = -0.2
    right_turn = 0.2
    
    #Declare min and max image rotation angles
    max_rot = 12
    min_rot = -12
    
    #Declare brightness parameters
    alpha = np.random.uniform(0.3, 1.0) 
    beta = 30 * np.random.uniform()
    
    #For center images, alter brigthness and randomly rotate.
    if img_type == 0:  #center images
        #Choose whether image is rotated or not
        rotate = random.randint(0, 1)
        
        #Alter image brighness
        img = (alpha*img + beta).astype(np.uint8)
        if rotate == 1:
            ang = np.random.uniform(min_rot, max_rot) #Select rotation angle: -12degrees < ang <12degrees
            M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
            img = cv2.warpAffine(img,M,(cols,rows))
            #if image rotate right (-ang), then steer right
            if ang < 0: steering = steering + (ang/min_rot*right_turn) #right_turn
            #if image rotates left (+ang), then steer left
            else: steering = steering + (ang/max_rot*left_turn) #left turn    
        else:
            #if the image is not rotated, keep the original image and steering angle
            ang = 0
            img = img
            steering = steering
            
    #For left or right images rotate image left or right, respectively.
    else:
        if img_type == 2:  # right images
            #For right images, rotate image left and add negative steer to the steering angle
            ang = np.random.uniform(0, max_rot)
            M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
            img = cv2.warpAffine(img,M,(cols,rows))
            #steering = steering + (left_turn/max_rot*steering)
            steering = steering + (ang/max_rot*left_turn)
        elif img_type == 1:  # left images
            #For left images, rotate image right and add positive steer to the steering angle
            ang = np.random.uniform(min_rot, 0)
            M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
            img = cv2.warpAffine(img,M,(cols,rows))
            #steering = steering + (right_turn/max_rot*steering)
            steering = steering + (ang/min_rot*right_turn) 
    skew_img = img
    return skew_img, steering, ang


#==================================================================================================
#                             generator FUNCTION: Generates data in batches
#==================================================================================================
def generator(data, augment=True):    
    '''
    This allows us to generate data on the fly instead of loading the entire
    data set unto memory.
    
    input: data from pandas (df)
    output: generated data samples (images, steering angles)
    '''
    while True:
        index_array = np.random.permutation(data.index)
        #create an array of camera types
        camera = ['center', 'left', 'right']
    
        index_array = data.index
        #set batch size
        batch_size = 128
    
        #Taking batches of the entire training data with batch size = batch_size
        for batch in range(0, len(index_array), batch_size): 
            #create subset of batch indices from indices within the batch size range
            batch_index_array = index_array[batch:(batch + batch_size)]  
            
            #Image dimensions
            img_height = 160
            img_width = 320
            img_size = (img_width, img_height)
        
            #create empty arrays to store image and steering values
            steer_array = np.empty([0], dtype=np.float32)
            imgs = np.empty([0, img_height, img_width, 3], dtype=np.float32)
        
            # Read in and preprocess a batch of images
            for index in batch_index_array:
                # Randomly select a camera type
                camera_type_index = np.random.randint(len(camera)) if augment else 0
                camera_type = camera[camera_type_index]  #This returns a 'center', 'left', 'right' string
                steering = data.get_value(index, 'steering') 
            
                # Extract image filename from dataframe
                img_filename = os.path.join(dir, data.get_value(index, camera_type))      
                img = cv2.imread(img_filename)
            
                #convert from default openCV BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size, interpolation = cv2.INTER_AREA)
                
                #Augment training data only. augment = False for test data
                if augment:
                    img, steering, ang = Skew_Image(img, camera_type_index, steering)

                #Populate array with images and steering angles
                steer_array = np.append(steer_array, [steering])
                imgs = np.append(imgs, [img], axis=0)
                 
            yield (imgs, steer_array)
    
#==========================================================================
#                               KERAS MODEL
#=========================================================================
#Import all necessary keras libraries
from keras import models, optimizers, backend
from keras.models import Sequential, Model
from keras.layers import core, convolutional, pooling, Lambda, Cropping2D
from keras.callbacks import ModelCheckpoint

#Define callback function
callback = ModelCheckpoint('./Weights/weights_{epoch:02d}.hdf5', monitor='mean_squared_error', verbose=0, save_weights_only=True)


model = Sequential()

#Lambda layer for data normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

#Crop layer to remove unwanted parts of the image 
model.add(Cropping2D(cropping=((60,20), (0,0))))

#1st convolution layer with max pooling
model.add(convolutional.Convolution2D(8, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))

#2nd convolution layer with max pooling
model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))

#3rd convolution layer with max pooling
model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))

#Flatten layer
model.add(core.Flatten())

#Dense layer with relu and dropout layer
model.add(core.Dense(500, activation='relu'))
model.add(core.Dropout(.5))

#Dense layer with relu and dropout layer
model.add(core.Dense(100, activation='relu'))
model.add(core.Dropout(.25))

#Dense layer with relu and dropout layer
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))

model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')

model.fit_generator(
        generator(train_data),
        samples_per_epoch = 8000,
        nb_epoch = 5,
        verbose=1,
        validation_data = generator(valid_data, augment=False),
        callbacks = [callback],
        nb_val_samples = valid_data.shape[0],
    )

#Save the final model
model.save("./model.h5")

#Print out model summary
model.summary()