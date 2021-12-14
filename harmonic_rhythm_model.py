import pickle
import os
import numpy as np
import keras_metrics as km
from keras.layers import Input
from keras.layers  import concatenate
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
from config import *

def create_training_data(segment_length=SEGMENT_LENGTH, corpus_path=CORPUS_PATH, val_ratio=HAR_VAL_RATIO):

    # Load corpus
    with open('rhythm_'+corpus_path, "rb") as filepath:
        corpus = pickle.load(filepath)

    # Inputs and targets for the training set
    input_melody_left = []
    input_melody_mid = []
    input_melody_right = []
    input_beat_left = []
    input_beat_mid = []
    input_beat_right = []
    input_rhythm_left = []
    output_rhythm = []

    # Inputs and targets for the validation set
    input_melody_left_val = []
    input_melody_mid_val = []
    input_melody_right_val = []
    input_beat_left_val = []
    input_beat_mid_val = []
    input_beat_right_val = []
    input_rhythm_left_val = []
    output_rhythm_val = []

    cnt = 0
    np.random.seed(0)

    # Process each melody sequence in the corpus
    for idx, melody in enumerate(corpus[0]):
        
        # Randomly assigned to the training or validation set with the probability
        if np.random.rand()>val_ratio:

            train_or_val = 'train'
        
        else:

            train_or_val = 'val'

        # Load the corresponding beat and rhythm sequence
        beat = corpus[1][idx]
        rhythm = corpus[2][idx]

        # '131', '4', '3' for melody, beat and rhythm sequences of paddings respectively
        melody = [131]*segment_length + melody + [131]*segment_length
        beat = [4]*segment_length + beat + [4]*segment_length
        rhythm = [3]*segment_length + rhythm + [3]*segment_length
        
        # Create pairs
        for i in range(segment_length, len(melody)-segment_length):
            
            melody_left = melody[i-segment_length: i] 
            melody_mid = melody[i]
            melody_right = melody[i+1: i+segment_length+1][::-1]
            beat_left = beat[i-segment_length: i] 
            beat_mid = beat[i]
            beat_right = beat[i+1: i+segment_length+1][::-1]
            rhythm_left = rhythm[i-segment_length: i]
            target = rhythm[i]
                     
            if train_or_val=='train':

                input_melody_left.append(melody_left)
                input_melody_mid.append(melody_mid)
                input_melody_right.append(melody_right)
                input_beat_left.append(beat_left)
                input_beat_mid.append(beat_mid)
                input_beat_right.append(beat_right)
                input_rhythm_left.append(rhythm_left)
                output_rhythm.append(target)
            
            else:

                input_melody_left_val.append(melody_left)
                input_melody_mid_val.append(melody_mid)
                input_melody_right_val.append(melody_right)
                input_beat_left_val.append(beat_left)
                input_beat_mid_val.append(beat_mid)
                input_beat_right_val.append(beat_right)
                input_rhythm_left_val.append(rhythm_left)
                output_rhythm_val.append(target)

        cnt += 1

    print("Successfully read %d pieces" %(cnt))

    # One-hot vectorization
    input_melody_left = to_categorical(input_melody_left, num_classes=132)
    input_melody_mid = to_categorical(input_melody_mid, num_classes=132)
    input_melody_right = to_categorical(input_melody_right, num_classes=132)
    input_beat_left = to_categorical(input_beat_left, num_classes=5)
    input_beat_mid = to_categorical(input_beat_mid, num_classes=5)
    input_beat_right = to_categorical(input_beat_right, num_classes=5)
    input_rhythm_left = to_categorical(input_rhythm_left, num_classes=4)
    output_rhythm = to_categorical(output_rhythm, num_classes=4)
    
    if len(input_melody_left_val)!=0:

        input_melody_left_val = to_categorical(input_melody_left_val, num_classes=132)
        input_melody_mid_val = to_categorical(input_melody_mid_val, num_classes=132)
        input_melody_right_val = to_categorical(input_melody_right_val, num_classes=132)
        input_beat_left_val = to_categorical(input_beat_left_val, num_classes=5)
        input_beat_mid_val = to_categorical(input_beat_mid_val, num_classes=5)
        input_beat_right_val = to_categorical(input_beat_right_val, num_classes=5)
        input_rhythm_left_val = to_categorical(input_rhythm_left_val, num_classes=4)
        output_rhythm_val = to_categorical(output_rhythm_val, num_classes=4)
    
    return (input_melody_left, input_melody_mid, input_melody_right, input_beat_left, input_beat_mid, input_beat_right, input_rhythm_left, output_rhythm), \
           (input_melody_left_val, input_melody_mid_val, input_melody_right_val, input_beat_left_val, input_beat_mid_val, input_beat_right_val, input_rhythm_left_val, output_rhythm_val)


def build_rhythm_model(segment_length, rnn_size, num_layers, weights_path=None):

    # Create input layer
    input_melody_left = Input(shape=(segment_length, 132), 
                        name='input_melody_left')
    input_melody_mid = Input(shape=(132, ), 
                        name='input_melody_mid')
    input_melody_right = Input(shape=(segment_length, 132), 
                        name='input_melody_right')
    input_beat_left = Input(shape=(segment_length, 5), 
                        name='input_beat_left')
    input_beat_mid = Input(shape=(5, ), 
                        name='input_beat_mid')
    input_beat_right = Input(shape=(segment_length, 5), 
                        name='input_beat_right')
    input_rhythm_left = Input(shape=(segment_length, 4), 
                        name='input_rhythm_left')

    # Update variable names
    melody_left = input_melody_left
    melody_right = input_melody_right
    beat_left = input_beat_left
    beat_right = input_beat_right
    rhythm_left = input_rhythm_left

    return_sequences = True

    # Creating the hidden layer of the LSTM
    for idx in range(num_layers):

        if idx == num_layers - 1:

            return_sequences = False

        melody_left = LSTM(units=rnn_size, 
                           return_sequences=return_sequences,
                           name='melody_left_'+str(idx+1))(melody_left)
        
        melody_right = LSTM(units=rnn_size, 
                            return_sequences=return_sequences,
                            name='melody_right_'+str(idx+1))(melody_right)

        beat_left = LSTM(units=rnn_size, 
                         return_sequences=return_sequences,
                         name='beat_left_'+str(idx+1))(beat_left)
        
        beat_right = LSTM(units=rnn_size, 
                          return_sequences=return_sequences,
                          name='beat_right_'+str(idx+1))(beat_right)

        rhythm_left = LSTM(units=rnn_size, 
                           return_sequences=return_sequences,
                           name='rhythm_left_'+str(idx+1))(rhythm_left)

    # Create Dense hidden layers
    melody_mid = Dense(units=rnn_size,
                       activation='relu',
                       name='melody_mid')(input_melody_mid)
    melody_mid = BatchNormalization()(melody_mid)

    beat_mid = Dense(units=rnn_size,
                     activation='relu',
                     name='beat_mid')(input_beat_mid)
    beat_mid = BatchNormalization()(beat_mid)

    # Merge hidden layer output
    merge = concatenate(
        [
            melody_left,
            melody_mid,
            melody_right,
            beat_left,
            beat_mid,
            beat_right,
            rhythm_left
        ]
    )                    
    
    # Linear transformation of the merged vector
    merge= Dense(units=rnn_size,
                 activation='relu',
                 name='merge')(merge)
    prediction = BatchNormalization()(merge)

    # Create output layer
    output_layer = Dense(units=4, 
                         activation="softmax", 
                         name='output_layer')(prediction)

    model = Model(
                  inputs=[
                          input_melody_left,
                          input_melody_mid,
                          input_melody_right,
                          input_beat_left,
                          input_beat_mid,
                          input_beat_right,
                          input_rhythm_left
                         ],
                  outputs=output_layer
                 )

    model.compile(optimizer='rmsprop',
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy', km.f1_score()])
    
    if weights_path==None:

        model.summary()

    else:

        model.load_weights(weights_path)

    return model


def train_model(data,
                data_val, 
                segment_length=SEGMENT_LENGTH, 
                rnn_size=HAR_RNN_SIZE, 
                num_layers=HAR_NUM_LAYERS, 
                batch_size=HAR_BATCH_SIZE, 
                epochs=HAR_EPOCHS, 
                verbose=1,
                weights_path='rhythm_'+WEIGHTS_PATH):

    model = build_rhythm_model(segment_length, rnn_size, num_layers)

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

    if len(data_val[0])!=0:

        # With validation set
        history = model.fit(x={'input_melody_left': np.array(data[0]), 
                                'input_melody_mid': np.array(data[1]), 
                                'input_melody_right': np.array(data[2]), 
                                'input_beat_left': np.array(data[3]), 
                                'input_beat_mid': np.array(data[4]), 
                                'input_beat_right': np.array(data[5]), 
                                'input_rhythm_left': np.array(data[6])},
                            y=np.array(data[7]),
                            validation_data=({'input_melody_left': np.array(data_val[0]), 
                                               'input_melody_mid': np.array(data_val[1]), 
                                               'input_melody_right': np.array(data_val[2]), 
                                               'input_beat_left': np.array(data_val[3]), 
                                               'input_beat_mid': np.array(data_val[4]), 
                                               'input_beat_right': np.array(data_val[5]), 
                                               'input_rhythm_left': np.array(data_val[6])}, 
                                                data_val[7]),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[checkpoint])
    else:

        # Without validation set
        history = model.fit(x={'input_melody_left': np.array(data[0]), 
                               'input_melody_mid': np.array(data[1]), 
                               'input_melody_right': np.array(data[2]), 
                               'input_beat_left': np.array(data[3]), 
                               'input_beat_mid': np.array(data[4]), 
                               'input_beat_right': np.array(data[5]), 
                               'input_rhythm_left': np.array(data[6])},
                            y=np.array(data[7]),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[checkpoint])
    
    return history


if __name__ == "__main__":

    # Load the training and validation sets
    data, data_val = create_training_data()
    
    # Train model
    history = train_model(data, data_val)