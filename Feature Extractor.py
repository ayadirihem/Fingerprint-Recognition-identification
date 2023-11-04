#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D

import keras.backend as k
from keras.layers import Lambda

import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.image as mpimg
import cv2 as cv

# # 1. Splitting Data into training and validation

# In[2]:


POS_PATH = 'Data/Real'
NEG_PATH = 'Data/Fake'


# In[ ]:


# Moving Fake images to fake file
for directory in os.listdir('./SOCOFing/SOCOFing/Altered'):
    for file in os.listdir(os.path.join('./SOCOFing/SOCOFing/Altered',directory)):
        EX_PATH = os.path.join('./SOCOFing/SOCOFing/Altered',directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH,NEW_PATH)


# In[ ]:


# Moving Real images to Real file
for file in os.listdir('./SOCOFing/Real'):
    EX_PATH = os.path.join('./SOCOFing/Real',file)
    NEW_PATH = os.path.join(POS_PATH, file)
    os.replace(EX_PATH,NEW_PATH)



# In[ ]:
    
# print example of fake and real fingerprint
# Get a list of image filenames in each folder
print("****sample of data****")
fake_images = os.listdir('./Data/Fake')
real_images = os.listdir('./Data/Real')
images = []
titles = []
# Display images from the "fake" folder
for i, fake_image in enumerate(fake_images):
    img = cv.imread('./Data/Fake/'+fake_image)
    images.append(img)
    titles.append("falsifiée")
    if i == 2: break

for i, real_image in enumerate(real_images):
    img = cv.imread('./Data/Real/'+real_image)
    images.append(img)
    titles.append("réelle")
    if i == 3: break
for i in range(6):
        plt.subplot(6,6,i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

plt.show()  

# In[ ]:


import splitfolders

input_folder = "./Data" #Enter Input Folder
output = "./Fingerprint_DS" #Enter Output Folder

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(0.7,0.2,0.1))


# In[3]:


base_dir = 'Fingerprint_DS'


# In[4]:


train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


# In[5]:


# checking everything was fine
paths = [train_dir, val_dir]


# In[6]:


datagen = ImageDataGenerator(rescale=1./255)


# In[7]:


No_of_files_train = len(os.listdir(train_dir+'\Real')) + len(os.listdir(train_dir+'/Fake'))
No_of_files_val = len(os.listdir(val_dir+'/Real')) + len(os.listdir(val_dir+'/Fake'))
No_of_files_test = len(os.listdir(val_dir+'/Real')) + len(os.listdir(val_dir+'/Fake'))


# In[8]:


print("\n *****************************",
          "\n Total images: ",No_of_files_train+No_of_files_val,
          '\n Training: ', No_of_files_train,
          '\n Validation: ', No_of_files_val,
          '\n Testing: ', No_of_files_test,
          '\n *****************************')


# # 2.Feature Extraction

# In[10]:


datagen = ImageDataGenerator(rescale=1./255)
batch_size = 50



# In[11]:


# Create Feature Extractor model using vgg16
model_vgg16 = VGG16(weights="imagenet", include_top="false")
model_vgg16.summary()


# In[12]:


FeatureExtractor_model = Model(inputs=model_vgg16.inputs, outputs=model_vgg16.layers[-5].output)
FeatureExtractor_model.summary()


# In[13]:


def Feature_Extractor(path, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(path, target_size=(224, 224),
    batch_size=batch_size, class_mode='binary')
    print("Class indices:", generator.class_indices)
    i=0
    for inputs_batch, labels_batch in generator:
        features_batch = FeatureExtractor_model.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


# In[14]:

with tf.device('/CPU:0'):
    train_Features, train_labels = Feature_Extractor(train_dir,No_of_files_train)


# In[15]:

with tf.device('/CPU:0'):
    val_features, val_labels = Feature_Extractor(val_dir,No_of_files_val)


# # 3. Classification Model *


# In[ ]:


input_shape = (7, 7, 512)



# In[19]:
    
train_features = np.reshape(train_Features, (No_of_files_train, 7*7* 512))
validation_features = np.reshape(val_features, (No_of_files_val, 7*7* 512))

from keras import models
from keras import layers
from keras import optimizers
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', 
                                            patience=5,
                                            restore_best_weights=True)
model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), 
    loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_features, train_labels, epochs=30,
                    batch_size=20, 
                    validation_data=(validation_features, val_labels),
                    callbacks=[callback])



model.summary()

# In[20]:
def Feature_Extractor_Img(path):
    features = np.zeros(shape=(1, 7, 7, 512))
    generator = datagen.flow_from_directory(path, target_size=(224, 224),
    batch_size=batch_size, class_mode='binary')
    print("Class indices:", generator.class_indices)
    i=0
    for inputs_batch in generator:
        features_batch = FeatureExtractor_model.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
    return features


# In[21]:
    
from tensorflow.keras.applications.vgg16 import preprocess_input

def extract_features_from_single_image(image_path):
    # Load the image
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)  # Apply appropriate preprocessing
    
    # Reshape and extract features
    image_features = FeatureExtractor_model.predict(np.expand_dims(image_array, axis=0))
    
    return image_features

# Provide the path to the image you want to predict
image_path = 'Real/1__M_Right_index_finger.BMP'

# Extract features from the single image
single_image_features = extract_features_from_single_image(image_path)

# Reshape the features to match the expected input shape of your prediction model
reshaped_features = np.reshape(single_image_features, (1, 7 * 7 * 512))

# Use the trained model to predict using the extracted features
prediction = model.predict(reshaped_features)

# Convert the prediction probability to a binary label
predicted_label = "Real" if prediction > 0.5 else "Fake"

# Print the prediction
print("Predicted pourcentage:", prediction)
print("Predicted Label:", predicted_label)


# In[21]:
    
import pandas as pd

test_values = pd.read_csv('./testing/_annotations.csv')

test_values.drop(columns=['width','height','xmin','ymin','xmax','ymax'],inplace=True)

Values_predicted = []

plt.figure(figsize=(15,15))

for index in test_values.index:
        plt.subplot(5,5, index+1)
        # Provide the path to the image you want to predict
        image_path = './testing/'+test_values['filename'][index]

        # Extract features from the single image
        single_image_features = extract_features_from_single_image(image_path)

        # Reshape the features to match the expected input shape of your prediction model
        reshaped_features = np.reshape(single_image_features, (1, 7 * 7 * 512))

        # Use the trained model to predict using the extracted features
        prediction = model.predict(reshaped_features)

        # Convert the prediction probability to a binary label
        predicted_label = "Real" if prediction > 0.5 else "Fake"
        img = load_img(image_path)
        plt.imshow(img)
        plt.tight_layout(pad=8.0)
        plt.title(f"Class:{test_values['class'][index]}|Predicted:{predicted_label}|{prediction[0][0]*100}% ", pad=20)
        plt.axis('off')


# In[21]:
print("nnModel Evaluationn")
from sklearn.metrics import f1_score, precision_score, recall_score

with tf.device('/CPU:0'):
    test_Features, test_labels = Feature_Extractor(test_dir,50)

test_Features = np.reshape(test_Features, (50, 7*7* 512))

y_pred = model.predict(test_Features)
score3 = model.evaluate(test_Features, test_labels, verbose=1)

# Convert predicted probabilities to binary predictions (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate F1 score, precision, and recall
f1 = f1_score(test_labels, y_pred_binary)
precision = precision_score(test_labels, y_pred_binary)
recall = recall_score(test_labels, y_pred_binary)

print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print('CNN Model Test accuracy:', score3[1])

model.save('saved_model/my_model')
FeatureExtractor_model.save('saved_model/FeatureExtractor_model')

# In[21]:
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# In[21]:
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# In[21]:

new_model = tf.keras.models.load_model('saved_model/my_model')


import pandas as pd

test_values = pd.read_csv('./testing/_annotations.csv')

test_values.drop(columns=['width','height','xmin','ymin','xmax','ymax'],inplace=True)

Values_predicted = []

plt.figure(figsize=(15,15))

for index in test_values.index:
        plt.subplot(5,5, index+1)
        # Provide the path to the image you want to predict
        image_path = './testing/'+test_values['filename'][index]

        # Extract features from the single image
        single_image_features = extract_features_from_single_image(image_path)

        # Reshape the features to match the expected input shape of your prediction model
        reshaped_features = np.reshape(single_image_features, (1, 7 * 7 * 512))

        # Use the trained model to predict using the extracted features
        prediction = new_model.predict(reshaped_features)

        # Convert the prediction probability to a binary label
        predicted_label = "Real" if prediction > 0.5 else "Fake"
        img = load_img(image_path)
        plt.imshow(img)
        plt.tight_layout(pad=8.0)
        plt.title(f"Class:{test_values['class'][index]}|Predicted:{predicted_label}", pad=20)
        plt.axis('off')
        

# In[21]:
        
        # print example of fake and real fingerprint
        # Get a list of image filenames in each folder
        print("****sample of data****")
        fake_images = os.listdir('./Fingerprint_DS/test/Fake')
        real_images = os.listdir('./Fingerprint_DS/test/Real')
        images = []
        titles = []
        plt.figure(figsize=(15,15))
        # Display images from the "fake" folder
        for i, fake_image in enumerate(fake_images):
            image_path = './Fingerprint_DS/test/Fake/'+fake_image
            # Extract features from the single image
            single_image_features = extract_features_from_single_image(image_path)

            # Reshape the features to match the expected input shape of your prediction model
            reshaped_features = np.reshape(single_image_features, (1, 7 * 7 * 512))

            # Use the trained model to predict using the extracted features
            prediction = new_model.predict(reshaped_features)

            # Convert the prediction probability to a binary label
            predicted_label = "Real" if prediction > 0.5 else "Fake"
            img = load_img(image_path)
            images.append(img)
            titles.append(f"Class:Fake|Predicted:{predicted_label}  ")
            if i == 2: break

        for i, real_image in enumerate(real_images):
            image_path = './Fingerprint_DS/test/Real/'+real_image
            # Extract features from the single image
            single_image_features = extract_features_from_single_image(image_path)

            # Reshape the features to match the expected input shape of your prediction model
            reshaped_features = np.reshape(single_image_features, (1, 7 * 7 * 512))

            # Use the trained model to predict using the extracted features
            prediction = new_model.predict(reshaped_features)

            # Convert the prediction probability to a binary label
            predicted_label = "Real" if prediction > 0.5 else "Fake"
            img = load_img(image_path)
            images.append(img)
            titles.append(f"Class:Real|Predicted:{predicted_label} ")
            if i == 3: break
        for i in range(6):
                plt.subplot(6,6,i+1)
                plt.imshow(images[i])
                plt.title(titles[i])
                plt.xticks([])
                plt.yticks([])
        plt.tight_layout(pad=8.0)
        plt.show()  
