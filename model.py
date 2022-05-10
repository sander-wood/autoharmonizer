import os
import math
import pickle
import numpy as np
import keras_metrics as km
from config import *
from tqdm import trange
from keras import Model
from keras.utils import Sequence
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers  import concatenate
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical

class DataGenerator(Sequence):
 
    def __init__(self, 
                 input_melody_left, 
                 input_melody_right, 
                 input_beat_left, 
                 input_beat_right, 
                 input_chord_left,
                 output_chord,
                 chord_nums,
                 batch_size=BATCH_SIZE, 
                 shuffle=True):
        self.input_melody_left = input_melody_left
        self.input_melody_right = input_melody_right
        self.input_beat_left = input_beat_left
        self.input_beat_right = input_beat_right
        self.input_chord_left = input_chord_left
        self.output_chord = output_chord
        self.chord_nums = chord_nums
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return math.ceil(len(self.input_melody_left) / self.batch_size)
    
    def __getitem__(self, index):
        onehot_melody_left = to_categorical(self.input_melody_left[index*self.batch_size:(index+1)*self.batch_size], num_classes=128)
        onehot_melody_right = to_categorical(self.input_melody_right[index*self.batch_size:(index+1)*self.batch_size], num_classes=128)
        onehot_beat_left = to_categorical(self.input_beat_left[index*self.batch_size:(index+1)*self.batch_size], num_classes=5)
        onehot_beat_right = to_categorical(self.input_beat_right[index*self.batch_size:(index+1)*self.batch_size], num_classes=5)
        onehot_chord_left = to_categorical(self.input_chord_left[index*self.batch_size:(index+1)*self.batch_size], num_classes=self.chord_nums)
        onehot_chord = to_categorical(self.output_chord[index*self.batch_size:(index+1)*self.batch_size], num_classes=self.chord_nums)

        return ([onehot_melody_left, onehot_melody_right, onehot_beat_left, onehot_beat_right, onehot_chord_left], onehot_chord)


def create_training_data(segment_length=SEGMENT_LENGTH, chord_types_path=CHORD_TYPES_PATH, corpus_path=CORPUS_PATH,  val_ratio=VAL_RATIO):

     # Load corpus
    with open(corpus_path, "rb") as filepath:
        data_corpus = pickle.load(filepath)
    # Load chord types
    with open(chord_types_path, "rb") as filepath:
        chord_types = pickle.load(filepath)

    # mapping chord name to int
    chord_types_dict = {chord_types[i]: i for i in range(len(chord_types))}

    # Inputs and targets for the training set
    input_melody_left = []
    input_melody_right = []
    input_beat_left = []
    input_beat_right = []
    input_chord_left = []
    output_chord = []

    # Inputs and targets for the validation set
    input_melody_left_val = []
    input_melody_right_val = []
    input_beat_left_val = []
    input_beat_right_val = []
    input_chord_left_val = []
    output_chord_val = []

    cnt = 0
    np.random.seed(0)

    # Process each song sequence in the corpus
    for songs_idx in trange(len(data_corpus)):
        
        songs = data_corpus[songs_idx]

        # Randomly assigned to the training or validation set with the probability
        if np.random.rand()>val_ratio:
            train_or_val = 'train'
        
        else:
            train_or_val = 'val'

        for song in songs:
            # Load the corresponding beat and rhythm sequence
            song_melody = segment_length*[0] + song[0] + segment_length*[0]
            song_beat = segment_length*[0] + song[1] + segment_length*[0]
            song_chord = segment_length*[0] + [chord_types_dict[cho] for cho in song[2]] + segment_length*[0]

            # Create pairs
            for idx in range(segment_length, len(song_melody)-segment_length):
                
                melody_left = song_melody[idx-segment_length:idx]
                melody_right = song_melody[idx:idx+segment_length][::-1]
                beat_left = song_beat[idx-segment_length:idx]
                beat_right = song_beat[idx:idx+segment_length][::-1]
                chord_left = song_chord[idx-segment_length:idx]
                chord = song_chord[idx]

                if train_or_val=='train':
                    input_melody_left.append(melody_left)
                    input_melody_right.append(melody_right)
                    input_beat_left.append(beat_left)
                    input_beat_right.append(beat_right)
                    input_chord_left.append(chord_left)
                    output_chord.append(chord)
                
                else:
                    input_melody_left_val.append(melody_left)
                    input_melody_right_val.append(melody_right)
                    input_beat_left_val.append(beat_left)
                    input_beat_right_val.append(beat_right)
                    input_chord_left_val.append(chord_left)
                    output_chord_val.append(chord)

        cnt += 1

    print("Successfully read %d pieces" %(cnt))
     
    return (input_melody_left, input_melody_right, input_beat_left, input_beat_right, input_chord_left, output_chord), \
           (input_melody_left_val, input_melody_right_val, input_beat_left_val, input_beat_right_val, input_chord_left_val, output_chord_val)


def build_model(segment_length, rnn_size, num_layers, dropout, weights_path=None, chord_types_path=CHORD_TYPES_PATH, training=True):

    # Load chord types
    with open(chord_types_path, "rb") as filepath:
        chord_types = pickle.load(filepath)

    # Create input layer with embedding
    input_melody_left = Input(shape=(segment_length, 128), 
                              name='input_melody_left')
    melody_left = TimeDistributed(Dense(12, activation='relu'))(input_melody_left)

    input_melody_right = Input(shape=(segment_length, 128),
                                 name='input_melody_right')
    melody_right = TimeDistributed(Dense(12, activation='relu'))(input_melody_right)

    input_beat_left = Input(shape=(segment_length, 5),
                            name='input_beat_left')
    beat_left = TimeDistributed(Dense(2, activation='relu'))(input_beat_left)

    input_beat_right = Input(shape=(segment_length, 5),
                                name='input_beat_right')
    beat_right = TimeDistributed(Dense(2, activation='relu'))(input_beat_right)

    input_chord_left = Input(shape=(segment_length, len(chord_types)),
                                name='input_chord_left')
    chord_left = TimeDistributed(Dense(int(math.sqrt(len(chord_types))), activation='relu'))(input_chord_left)

    return_sequences = True

    # Creating the hidden layer of the LSTM
    for idx in range(num_layers):

        if idx==num_layers-1:
            return_sequences = False

        melody_left = LSTM(rnn_size, 
                           name='melody_left_'+str(idx+1),
                           return_sequences=return_sequences)(melody_left)
        if idx!=num_layers-1:
            melody_left = TimeDistributed(Dense(rnn_size, activation='relu'))(melody_left)
        melody_left = BatchNormalization()(melody_left)
        melody_left = Dropout(dropout)(melody_left, training=training)

        melody_right = LSTM(rnn_size,
                            name='melody_right_'+str(idx+1),
                            return_sequences=return_sequences)(melody_right)
        if idx!=num_layers-1:
            melody_right = TimeDistributed(Dense(rnn_size, activation='relu'))(melody_right)
        melody_right = BatchNormalization()(melody_right)
        melody_right = Dropout(dropout)(melody_right, training=training)

        beat_left = LSTM(rnn_size,
                            name='beat_left_'+str(idx+1),
                            return_sequences=return_sequences)(beat_left)
        if idx!=num_layers-1:
            beat_left = TimeDistributed(Dense(rnn_size, activation='relu'))(beat_left)
        beat_left = BatchNormalization()(beat_left)
        beat_left = Dropout(dropout)(beat_left, training=training)

        beat_right = LSTM(rnn_size,
                            name='beat_right_'+str(idx+1),
                            return_sequences=return_sequences)(beat_right)
        if idx!=num_layers-1:
            beat_right = TimeDistributed(Dense(rnn_size, activation='relu'))(beat_right)
        beat_right = BatchNormalization()(beat_right)
        beat_right = Dropout(dropout)(beat_right, training=training)

        chord_left = LSTM(rnn_size,
                            name='chord_left_'+str(idx+1),
                            return_sequences=return_sequences)(chord_left)
        if idx!=num_layers-1:
            chord_left = TimeDistributed(Dense(rnn_size, activation='relu'))(chord_left)
        chord_left = BatchNormalization()(chord_left)
        chord_left = Dropout(dropout)(chord_left, training=training)

    # Merge hidden layer output
    merge = concatenate(
                        [
                         melody_left,
                         melody_right,
                         beat_left,
                         beat_right,
                         chord_left
                        ]
                       )        
    # Creating the hidden layer of the LSTM
    for idx in range(num_layers):
        merge= Dense(units=rnn_size,
                    activation='relu',
                    name='merge'+str(idx))(merge)
        merge = BatchNormalization()(merge)
        merge = Dropout(dropout)(merge, training=training)

    # Create output layer
    output_chord = Dense(units=len(chord_types),
                     activation='softmax',
                     name='output_chord')(merge)
    model = Model(
                  inputs=[
                          input_melody_left,
                          input_melody_right,
                          input_beat_left,
                          input_beat_right,
                          input_chord_left
                          ],
                  outputs= output_chord
                 )

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', km.f1_score()])
    
    if weights_path==None:
        model.summary()

    else:
        model.load_weights(weights_path)

    return model


def train_model(data, 
                data_val, 
                segment_length=SEGMENT_LENGTH, 
                rnn_size=RNN_SIZE, 
                num_layers=NUM_LAYERS, 
                dropout=DROPOUT,
                epochs=EPOCHS, 
                verbose=1,
                weights_path=WEIGHTS_PATH):

    with open(CHORD_TYPES_PATH, "rb") as filepath:
        chord_nums = len(pickle.load(filepath))
        
    model = build_model(segment_length, rnn_size, num_layers, dropout)

    # Load or remove existing weights
    if os.path.exists(weights_path):
        
        try:
            model.load_weights(weights_path)
            print("checkpoint loaded")
        
        except:
            os.remove(weights_path)
            print("checkpoint deleted")

    # Set monitoring indicator
    if len(data_val[0])!=0:
        monitor = 'val_loss'

    else:
        monitor = 'loss'
        
    # Save weights
    checkpoint = ModelCheckpoint(filepath=weights_path,
                                 monitor=monitor,
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')

    train_generator = DataGenerator(input_melody_left=data[0],
                                    input_melody_right=data[1],
                                    input_beat_left=data[2],
                                    input_beat_right=data[3],
                                    input_chord_left=data[4],
                                    output_chord=data[5],
                                    chord_nums=chord_nums)
        
    if len(data_val[0])!=0:

        val_generator = DataGenerator(input_melody_left=data_val[0],
                                      input_melody_right=data_val[1],
                                      input_beat_left=data_val[2],
                                      input_beat_right=data_val[3],
                                      input_chord_left=data_val[4],
                                      output_chord=data_val[5],
                                      chord_nums=chord_nums)

        # With validation set
        history = model.fit(x=train_generator,
                            validation_data=val_generator,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[checkpoint])
    
    else:
        
        # Without validation set
        history = model.fit(x=train_generator,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[checkpoint])

    return history


if __name__ == "__main__":

    # Load the training and validation sets
    data, data_val = create_training_data()
    
    # Train model
    history = train_model(data, data_val)