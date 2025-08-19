import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout ,GlobalAveragePooling2D
from keras.models import Sequential ##  sequential model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix,classification_report
import os, shutil
import warnings
warnings.filterwarnings('ignore')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

path = r"C:\Users\capri\Desktop\Facultate\ANUL1\SEMESTRUL 1\Machine learning\proiect 2\Vegetable_Images_test_train\\"

train_path = path + "train"
validation_path = path + "validation"
test_path = path + "test"

image_categories = os.listdir(train_path)

def plot_images(image_categories):
    
    # Create a figure
    plt.figure(figsize=(12, 12))
    for i, cat in enumerate(image_categories):
        # Load images for the ith category
        image_path = train_path + '/' + cat
        images_in_folder = os.listdir(image_path)
        first_image_of_folder = images_in_folder[i]
        first_image_path = image_path + '/' + first_image_of_folder
        img = image.load_img(first_image_path)
        img_arr = image.img_to_array(img)/255
        
        
        # Create Subplot and plot the images
        plt.subplot(4, 4, i+1)
        plt.imshow(img_arr)
        plt.title(cat)
        plt.axis('off')
        
    plt.show()

plot_images(image_categories)



images_length = {}
for i, cat in enumerate(image_categories):
    # Load images for the ith category
    image_path = train_path + '/' + cat
    images_in_folder = os.listdir(image_path)
    images_length[cat] = len(images_in_folder)

# Create a horizontal bar plot without the 'count' label
pd.DataFrame(images_length, index=['']).T.plot(kind='barh',color='green', legend=False)



# 1. Train Set
train_gen = ImageDataGenerator( featurewise_center=False,  
                                samplewise_center=False,  
                                featurewise_std_normalization=False,  
                                samplewise_std_normalization=False,  
                                zca_whitening=False,                
                                rotation_range=10,               
                                zoom_range = 0.1,                   
                                width_shift_range=0.2,  
                                height_shift_range=0.2, 
                                horizontal_flip=True,  
                                vertical_flip=False,
                              ) 
train_image_generator = train_gen.flow_from_directory(
                                            train_path,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='categorical')

# 2. Validation Set
val_gen = ImageDataGenerator() # Normalise the data
val_image_generator = val_gen.flow_from_directory(
                                            validation_path,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='categorical')

#3. Testing Set
test_gen = ImageDataGenerator() # Normalise the data
test_image_generator = test_gen.flow_from_directory(
                                            validation_path,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='categorical')


class_map = dict([(v, k) for k, v in train_image_generator.class_indices.items()])
print("\n",class_map)


## convolutional layers
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


## flatten layer
model.add(Flatten())

## fully connected layers
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(512,activation='relu'))

## output layer
model.add(Dense(15, activation = "softmax"))

model.summary()

#####################################################################


# Compile and fit the model
early_stopping = keras.callbacks.EarlyStopping(patience=3,monitor='val_loss',restore_best_weights=True) # Set up callbacks


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics='accuracy')
hist = model.fit(train_image_generator, 
                 epochs=25, 
                 verbose=1, 
                 validation_data=val_image_generator, 
                 steps_per_epoch = 15000//32, 
                 validation_steps = 3000//32, 
                 callbacks=early_stopping,
                 workers=4
                )

model.save('model_trained_25_epochs.h5')

def plot(hist):
    h = hist.history
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(h['loss']) + 1)  # Start x-axis counting from 1
    plt.plot(epochs, h['loss'], c='red', label='Training Loss')
    plt.plot(epochs, h['val_loss'], c='red', linestyle='--', label='Validation Loss')
    plt.plot(epochs, h['accuracy'], c='blue', label='Training Accuracy')
    plt.plot(epochs, h['val_accuracy'], c='blue', linestyle='--', label='Validation Accuracy')
    plt.xlabel("Number of Epochs")
    plt.legend(loc='best')
    plt.show()

plot(hist)


#########################################################
#Trebuie rulat si de la inceput pana la linia 135

model = tf.keras.models.load_model(r"C:\Users\capri\Desktop\Facultate\ANUL1\SEMESTRUL 1\Machine learning\proiect 2\model_trained_25_epochs.h5")

## helper function to get accuracy of the model
def eval_(model):
    test_loss, test_acc = model.evaluate(test_image_generator)
    return f'Model Test loss : {np.round(test_loss,2)} and Test Accuracy : {np.round(test_acc,2)} '

evaluation_result = eval_(model)
print(evaluation_result)



def predict(label,image_number):

    if label not in class_map.values() or image_number >=1000:
        print('Wrong Input 1. check if label in those 15 classes 2. image_number must be less than 1000')
    image_path = train_path + '/' + label
    images_in_folder = os.listdir(image_path)
    
    first_image_of_folder = images_in_folder[image_number]
    image_path += '/'+first_image_of_folder
    
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    test_img_input = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    
    true_label = label
    predicted_label = class_map[np.argmax(model.predict(test_img_input))]
    correct_prediction = true_label == predicted_label
    
    plt.imshow(img)
    plt.title(f'Predicted Label: {predicted_label}\nTrue Label: {true_label}\nCorrect Prediction: {correct_prediction}')
    plt.axis('off')
    plt.show()
    
    predicted_probs = model.predict(test_img_input)[0]
    plt.figure(figsize=(10, 5))
    plt.barh(list(class_map.values()), predicted_probs, color='blue')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Class Labels')
    plt.title(f'True Label: {true_label}')
    plt.show()
    

predict('Potato',998)  
predict('Carrot',9)
predict('Bean',98)


def plot(hist):
    h = hist.history
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(h['loss']) + 1)  # Start x-axis counting from 1
    plt.plot(epochs, h['loss'], c='red', label='Training Loss')
    plt.plot(epochs, h['val_loss'], c='red', linestyle='--', label='Validation Loss')
    plt.plot(epochs, h['accuracy'], c='blue', label='Training Accuracy')
    plt.plot(epochs, h['val_accuracy'], c='blue', linestyle='--', label='Validation Accuracy')
    plt.xlabel("Number of Epochs")
    plt.legend(loc='best')

    # Find the epoch at which training stopped
    if 'val_accuracy' in h:
        best_epoch = np.argmax(h['val_accuracy']) + 1  # +1 to convert 0-based index to epoch number
        plt.title(f"Training stopped at epoch {best_epoch} with highest validation accuracy")
    else:
        plt.title("Training completed all epochs")

    plt.show()
plot(hist)
