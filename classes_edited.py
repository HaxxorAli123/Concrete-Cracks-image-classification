#%%
import os 
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
print(keras.backend.backend())
# %%
# 1. Import Packages
from keras import layers,losses,metrics,activations,optimizers,initializers,regularizers,callbacks,applications
from keras.utils import plot_model
import pydot
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import shutil
from collections import defaultdict
import random
from tensorflow.keras.models import load_model

#%%
# 2. Data Loading
os.getcwd()
#os.chdir(r"C:\Users\User\OneDrive\Desktop\YP\Subjects\Capstone\project3")
#os.getcwd()

#%%
def count_files_in_directory(directory_path):
    #list down all files in entry
    entries = os.listdir(directory_path)
    #filter entries to include only needed files
    files =[entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
    return len(files)

#%%
good_file_count = count_files_in_directory("datasets/Negative")
crack_file_count = count_files_in_directory("datasets/Positive")
print("Good file folder:",good_file_count,"\ncrack file folder:",crack_file_count)

#%%
def split_files(source_dir, train_dir, val_dir, test_dir,split_ratio):
    #checking if file exist in folder
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    #getting all files in directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f))]
    random.shuffle(files)


    train_split = int(split_ratio[0] * len(files))
    val_split = int(split_ratio[1] * len(files) + train_split)

    train_files = files[:train_split]
    val_files = files[train_split:val_split]
    test_files = files[val_split:]

    for file in train_files:
        shutil.move(os.path.join(source_dir,file),os.path.join(train_dir,file))

    for file in val_files:
        shutil.move(os.path.join(source_dir,file),os.path.join(val_dir,file))
    
    for file in test_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(test_dir, file))
    print(f"Files split successfully: {len(train_files)} files in {train_dir}, {len(val_files)} files in {val_dir}, {len(test_files)} files in {test_dir}.")

#%%
#good dataset (segregate between 70:30 ratio)
source_directory = 'datasets/Negative'
destination_train = 'datasets/crack_detector/crack_dataset_splited/train/Negative'
destination_val = 'datasets/crack_detector/crack_dataset_splited/val/Negative'
destination_test = 'datasets/crack_detector/crack_dataset_splited/test/Negative'

split_files(source_directory,destination_train,destination_val,destination_test,split_ratio=[0.7,0.2,0.1])

#%%
#bad dataset
source_directory = 'datasets/Positive'
destination_train = 'datasets/crack_detector/crack_dataset_splited/train/Positive'
destination_val = 'datasets/crack_detector/crack_dataset_splited/val/Positive'
destination_test = 'datasets/crack_detector/crack_dataset_splited/test/Positive'

split_files(source_directory,destination_train,destination_val,destination_test,split_ratio=[0.7,0.2,0.1])

#%%
#setting up the train and test path
train_path = 'datasets/crack_detector/crack_dataset_splited/train'
val_path = 'datasets/crack_detector/crack_dataset_splited/val'
test_path = 'datasets/crack_detector/crack_dataset_splited/test'

#%%
BATCH_SIZE = 10
IMAGE_SIZE = (32,32)
train_dataset = keras.utils.image_dataset_from_directory(train_path,shuffle=True,
                                                         batch_size=BATCH_SIZE,
                                                         image_size=IMAGE_SIZE
                                                         )
val_dataset = keras.utils.image_dataset_from_directory(val_path, shuffle=True,
                                                       batch_size=BATCH_SIZE,
                                                       image_size=IMAGE_SIZE
                                                       )
test_dataset = keras.utils.image_dataset_from_directory(test_path,shuffle=True,
                                                         batch_size=BATCH_SIZE,
                                                         image_size=IMAGE_SIZE
                                                         )

#%%
# 3. inspect the data by plotting some images
class_names = train_dataset.class_names
batch_1 = train_dataset.take(1)
for images,label in batch_1:
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[label[i]])
plt.show()

#%%
#  Create the data augmentation layers 
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomTranslation(0.3,0.3),
    layers.RandomZoom(0.2)
])

#%%
# Testing out image implementation layers
for images,label in train_dataset.take(1):
    first_image = images[0]
    plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0] / 255.0)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
plt.show()

#%%
# 4. Define the preprocess input layer
preprocess_input = applications.mobilenet_v2.preprocess_input

#%%
# 5. Perform transfer learning 
# (A) Create the feature extractor from the pre-trained model
IMG_SHAPE = IMAGE_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,
                                      weights='imagenet')
# Freeze the entire model
base_model.trainable = False
base_model.summary()

#%%
# (B) Classification Layers create
global_avg = layers.GlobalAveragePooling2D()
output_layers = layers.Dense(len(class_names),activation='softmax')

#%%
# (C) Using functional API to create the entire model pipeline
# a. begin with input
inputs = keras.Input(shape=IMG_SHAPE)
# b. Augmentation 
x = data_augmentation(inputs)
# c. Preprocessing
x = preprocess_input(x)
# d. feature extractor
x = base_model(x)
# e. classification layers
x = global_avg(x)
# x = layers.Dropout(0.5)(x) #over herrrrrreeeeeee
outputs = output_layers(x)
# f. Define the keras model 
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

#%%
# 5. Model compiling
optimizer = optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#%%
keras.utils.plot_model(model, to_file="static/mobilenet_v2.png",show_shapes=True)

#%%
# Evaluate the model b4 training 
print(model.evaluate(test_dataset))

#%%
# Directing the model to Tensorboard
log_dir = "logs/pre-trained-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = callbacks.EarlyStopping(patience=5,monitor='val_accuracy',min_delta=0.01,restore_best_weights=True)

#%%
# 6. Model training
EPOCHS = 2
history_first = model.fit(train_dataset,validation_data=val_dataset,epochs=EPOCHS,callbacks=[tensorboard_callback])

#%% Run tensorboard for pre-trained model
print(f"Run `tensorboard --logdir={log_dir}` to view logs in TensorBoard.")

#%%
# Evaluate the model after training
print(model.evaluate(test_dataset))

#%%
# 7. transfer learning 2nd phase
# (A) Unfreeze the entire base model
base_model.trainable = True
# (B) Freeze the earlier layers inside the base model
print(len(base_model.layers))  # Check total number of layers in the base model
fine_tune_at = len(base_model.layers) // 2 
print(fine_tune_at)
#%%
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True
base_model.summary()

# %%
log_dir_fined_tuned = "logs/fine-tuned-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_fined_tuned, histogram_freq=1)

#%%
# (C) Continue the training 
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#%%
# (D) Proceed the 2nd stage training
fine_tune_epoch = 4
total_epoch = EPOCHS + fine_tune_epoch
history_fine = model.fit(train_dataset,validation_data=val_dataset,epochs=total_epoch,
                         initial_epoch=EPOCHS,callbacks=[tensorboard_callback,early_stopping])

#%%
# Evaluate model again after 2nd training
print(model.evaluate(test_dataset))

#%% Run tensorboard for pre-trained model
print(f"Run `tensorboard --logdir={log_dir_fined_tuned}` to view logs in TensorBoard.")

#%%
#use the model to make predictions
for image_batch,label_batch in test_dataset.take(1):
    predictions = np.argmax(model.predict(image_batch),axis=1)
    predicted_class = [class_names[x] for x in predictions]
print(predicted_class)

#%%
# plot the graph to show image,prediction and label
plt.figure(figsize=(15,15))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_batch[i].numpy().astype('uint8'))
    plt.title(f"Prediction: {predicted_class[i]}, label: {class_names[label_batch[i]]}")
plt.show()

#%%
# saving model in a .h5 format file
os.makedirs("models",exist_ok=True)
model.save("models/mobilenet_v2.h5")


#%% loading model architecture
# loaded_model = load_model('models/mobilenet_v2.h5')
# loaded_model.summary()

#%% performance and reports
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Val Loss: {val_loss}")
print(f"Val Accuracy: {val_accuracy}")


# %%
