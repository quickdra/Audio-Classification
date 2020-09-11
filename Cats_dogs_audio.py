import numpy as np
import pandas as pd
import IPython.display as ipd
import soundfile
import tensorflow as tf
import copy
import gc
from sklearn.preprocessing import StandardScaler
import tables
import random
tf.random.set_seed(10)
import librosa
import time
import matplotlib.pyplot as plt


file_path = '' #File path for the folder where the audio files are located
csv = 'train_test_split.csv' #CSV file containing the names of the audio files
file_names = pd.read_csv(csv)
file_names = file_names.fillna(0) # A few columns had null values

#Creating MFCC features for every 50ms window with 50% overlap
#Also adds white noise to every window as data augmentation which doubles the dataset
#Other augmentations such as slowing down or fastening the audio were tried but the performance deteroitated
#Returns features for every window and label
def create_features(file_names, file_path):
  flag = 0
  columns = ['train_cat', 'train_dog']
  for col in columns:
    for file in file_names[col]:
      filefp = fp+file
      data, fs = soundfile.read(filefp)
      data = data.reshape(-1,1)
      wn = np.random.randn(len(data)).reshape(-1,1)
      data_wn = data + 0.0075*wn
      scaler.fit(data)
      data = scaler.transform(data)
      data_wn = scaler.transform(data_wn)
      #Creating MFCC features for 50ms(800 data points) window and 50% overlap(400 hop_length)
      mfcc = librosa.feature.mfcc(y = data.reshape(data.shape[0],), sr = fs, n_fft = 800, hop_length = 400)
      mfcc_wn = librosa.feature.mfcc(y = data_wn.reshape(data_wn.shape[0],), sr = fs, n_fft = 800, hop_length = 400)
      #Finally for each of the windows 20 features are extracted
      mfcc = mfcc.reshape(-1,20)
      mfcc_wn = mfcc_wn.reshape(-1,20)
      #Creating labels
      if col == 'train_cat':
        x = 0
      else:
        x = 1
      label = np.array([[x] for i in range(mfcc.shape[0])]).reshape(-1,1)
      label_wn = np.array([[x] for i in range(mfcc_wn.shape[0])]).reshape(-1,1)
      data = np.concatenate((data, data_wn))
      label = np.concatenate((label, label_wn))
      if flag == 0:
        #if reading the first file
        windows = data
        window_labels = label
        flag = 1
      else:
        windows = np.concatenate((windows, data)))
        window_labels = np.concatenate((window_labels))
  return windows, window_labels

      
def create_model():
  lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps = 25*(n_dog+n_cat-20)//128,decay_rate = 1)
  #A simple bidirectional NN proved to perform well
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences = True, activation = 'sigmoid'), input_shape = (20,40)))
  model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, activation = 'sigmoid')))
  model.add(tf.keras.layers.Dense(1))
  model.add(tf.keras.layers.Activation('sigmoid'))
  model.compile(loss = tf.keras.losses.BinaryCrossentropy(),optimizer = tf.keras.optimizers.Adam(lr_schedule), metrics = ['accuracy'])
  model.summary()
  return model

#Model_fp ---> file path where to save the model
def train_model(model, windows, window_labels, model_fp)
    #A sequence of 40 windows (1 second audio) is used as a data point
    #Hence the model can classify 1 sec of audio as either cat/dog audio
    dog_labels = l.root.data[n_cat:,:]
    dog_data = w.root.data[n_cat:,:].reshape(20, -1)
    index = np.random.randint(0,n_dog-40,n_dog-40)
    flag = 0
    for i in index:
        if flag == 0:
            dat1 = dog_data[:,i:i+40].reshape(1,20,40)
            lab1 = dog_labels[i]
            flag = 1
        else:
            dat1 = np.concatenate((dat1,dog_data[:,i:i+40].reshape(1,20,40)))
            lab1 = np.concatenate((lab1,dog_labels[i]))

    cat_labels = l.root.data[0:n_cat,:]
    cat_data = w.root.data[0:n_cat,:].reshape(20,-1)
    index = np.random.randint(0,n_cat-40, n_cat-40)
    flag = 0
    for i in index:
        if flag == 0:
            dat2 = cat_data[:,i:i+40].reshape(1,20,40)
            lab2 = cat_labels[i]
            flag = 1
        else:
            dat2 = np.concatenate((dat2,cat_data[:,i:i+40].reshape(1,20,40)))
            lab2 = np.concatenate((lab2,cat_labels[i]))

    new_dat = np.concatenate((dat1, dat2))
    new_labels = np.concatenate((lab1, lab2))
    index = [i for i in range(new_dat.shape[0])]
    random.shuffle(index)
    new_dat = new_dat[index]
    new_labels = new_labels[index]
    #t3 = time.time()
    #print("Time to get train data set up ",t3-t1)
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 10)
    model.fit(new_dat,new_labels, validation_split = 0.2, batch_size = 128, epochs = 100, callbacks = [callback])
    model.save(model_fp)
