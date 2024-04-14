#Importing Libraries 
import numpy as np
import pandas as pd
import os
import cv2 as cv
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import add, Conv2D,MaxPooling2D,UpSampling2D, Input,BatchNormalization, RepeatVector, Reshape
from keras.models import Model
np.random.seed(1)

#Mounting Google Drive and loading the Lol_Dataset
from google.colab import drive
drive.mount('/content/drive') 

InputPath="/content/drive/MyDrive/Graphics_dataset/high"

#Salt and Pepper noise function 

def addNoise(image, noise_level=0.05):
    # Copy the original image to avoid modifying it directly
    noise_added_image = np.copy(image)

    # Determine the number of noisy pixels
    num_noisy_pixels = int(noise_level * image.size)

    # Generate random indices for adding salt-and-pepper noise
    indices = np.random.choice(np.arange(image.size), size=num_noisy_pixels, replace=False)

    # Convert 1D indices to 2D coordinates
    salt_coords = np.unravel_index(indices[:num_noisy_pixels // 2], image.shape)
    pepper_coords = np.unravel_index(indices[num_noisy_pixels // 2:], image.shape)

    # Add salt noise
    noise_added_image[salt_coords] = 1

    # Add pepper noise
    noise_added_image[pepper_coords] = 0

    return noise_added_image




img = cv.imread(InputPath + "/100.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Display the original image
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original')

# Add noise to the original image
plt.subplot(1, 3, 2)
Noise = addNoise(img)
plt.imshow(Noise)
plt.title('With Noise')

# Convert the image to HSV and decrease the value channel
plt.subplot(1, 3, 3)
hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
hsv[..., 2] = hsv[..., 2] * 0.2
img1 = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

# Add noise to the modified image
Noise2 = addNoise(img1)
plt.imshow(Noise2)
plt.title('Modified with Noise')

plt.show()

#Preprocessing 
from tqdm import tqdm
HighPath = "/content/drive/MyDrive/Graphics_dataset/high"

def PreProcessData(ImagePath):
    X_=[]
    y_=[]
    count=0
    for imageName in tqdm(os.listdir(HighPath)):
        count=count+1
        low_img = cv.imread(HighPath + "/" + imageName)
        low_img = cv.cvtColor(low_img, cv.COLOR_BGR2RGB)
        low_img = cv.resize(low_img,(500,500))
        hsv = cv.cvtColor(low_img, cv.COLOR_BGR2HSV) #convert it to hsv
        hsv[...,2] = hsv[...,2]*0.2
        img_1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        Noisey_img = addNoise(img_1)
        X_.append(Noisey_img)
        y_.append(low_img)
    X_ = np.array(X_)
    y_ = np.array(y_)

    return X_,y_

X_,y_ = PreProcessData(InputPath)

#CNN Implementation-Instantiate Model
K.clear_session()
def InstantiateModel(in_):

    model_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_1)
    model_1 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_1)

    model_2 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_2 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)

    model_2_0 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)

    model_add = add([model_1,model_2,model_2_0])

    model_3 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_3)
    model_3 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3)

    model_3_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3_1 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3_1)

    model_3_2 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add)

    model_add_2 = add([model_3_1,model_3_2,model_3])

    model_4 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_2)
    model_4_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add)

    model_add_3 = add([model_4_1,model_add_2,model_4])

    model_5 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_3)
    model_5 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add_3)

    model_5 = Conv2D(3,(3,3), activation='relu',padding='same',strides=1)(model_5)

    return model_5

Input_Sample = Input(shape=(500, 500,3))
Output_ = InstantiateModel(Input_Sample)
Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)
Model_Enhancer.compile(optimizer="adam", loss='mean_squared_error')
Model_Enhancer.summary()

from keras.utils import plot_model
from IPython.display import Image

# Plot the Keras model and save it as an image
plot_model(Model_Enhancer, to_file='model.png', show_shapes=True, show_layer_names=True)

# Display the image
Image('model.png')



#Generate Input
def GenerateInputs(X,y):
    for i in range(len(X)):
        X_input = X[i].reshape(1,500,500,3)
        y_input = y[i].reshape(1,500,500,3)
        yield (X_input,y_input)
# Assuming Model_Enhancer is your Functional model
Model_Enhancer.fit(GenerateInputs(X_, y_), epochs=53, verbose=1, steps_per_epoch=8, shuffle=True)

TestPath ="/content/drive/MyDrive/Graphics_dataset/high"

#Extraction and Evalutaion
def ExtractTestInput(ImagePath):
    img = cv.imread(ImagePath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_ = cv.resize(img,(500,500))
    hsv = cv.cvtColor(img_, cv.COLOR_BGR2HSV) #convert it to hsv
    hsv[...,2] = hsv[...,2]*0.2
    img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    Noise = addNoise(img1)
    Noise = Noise.reshape(1,500,500,3)
    return Noise

    ImagePath=TestPath+"/13.png"

plt.figure(figsize=(30,30))
plt.subplot(5,5,1)
img_1 = cv.imread(ImagePath)
img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
img_1 = cv.resize(img_1, (500, 500))
plt.title("Ground Truth",fontsize=20)
plt.imshow(img_1)

plt.subplot(5,5,1+1)
img_ = ExtractTestInput(ImagePath)
img_ = img_.reshape(500,500,3)
plt.title("Low Light Image",fontsize=20)
plt.imshow(img_)

plt.subplot(5,5,1+2)
image_for_test = ExtractTestInput(ImagePath)
Prediction = Model_Enhancer.predict(image_for_test)
Prediction = Prediction.reshape(500,500,3)
img_[:,:,:] = Prediction[:,:,:]
plt.title("Enhanced Image",fontsize=20)
plt.imshow(img_)