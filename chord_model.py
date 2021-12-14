import pickle
import os
import numpy as np
from keras.layers import Input
from keras.layers  import concatenate
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from config import *

def mul_seg(sequence, idx, segments_length, direction, sep):

    segs = []

    if direction=='left':

        for i in range(-segments_length,0):

            segs += sequence[idx+i]

            # Add separator
            if i!=-1:

                segs.append(sep)
    
    elif direction=='right':

        for i in range(1,segments_length+1):

            segs += sequence[idx+i]

            # Add separator
            if i!=segments_length:

                segs.append(sep)

    return segs
        
        
def create_training_data(segments_length=SEGMENTS_LENGTH, corpus_path=CORPUS_PATH,  val_ratio=CHO_VAL_RATIO):

    # Load corpus
    with open('chord_'+corpus_path, "rb") as filepath:
        corpus = pickle.load(filepath)

    # Inputs and targets for the training set
    input_melody_left = []
    input_melody_mid = []
    input_melody_right = []
    input_chord_left = []
    output_chord = []

    # Inputs and targets for the validation set
    input_melody_left_val = []
    input_melody_mid_val = []
    input_melody_right_val = []
    input_chord_left_val = []
    output_chord_val = []

    cnt = 0
    np.random.seed(0)

    # Process each chord sequence in the corpus
    for idx, chord in enumerate(corpus[1]):
        
        # Randomly assigned to the training or validation set with the probability
        if np.random.rand()>val_ratio:

            train_or_val = 'train'
        
        else:

            train_or_val = 'val'

        # Load the corresponding melody sequence
        melody = corpus[0][idx]
        
        # '131', '14' for melody and chord sequences of paddings respectively
        for i in range(segments_length):

            melody.insert(0,[131]*16)
            melody.append([131]*16)
            chord.insert(0,[14]*4)

        # Create pairs
        for i in range(segments_length, len(chord)):
            
            # '131', '15' for melody and chord sequences of separators respectively
            melody_left = mul_seg(melody,i,segments_length,'left',132)
            melody_mid = melody[i]
            melody_right = mul_seg(melody,i,segments_length,'right',132)[::-1]
            chord_left = mul_seg(chord,i,segments_length,'left',15)
            target = chord[i]

            if train_or_val=='train':

                input_melody_left.append(melody_left)
                input_melody_mid.append(melody_mid)
                input_melody_right.append(melody_right)
                input_chord_left.append(chord_left)
                output_chord.append(target)
            
            else:

                input_melody_left_val.append(melody_left)
                input_melody_mid_val.append(melody_mid)
                input_melody_right_val.append(melody_right)
                input_chord_left_val.append(chord_left)
                output_chord_val.append(target)

        cnt += 1

    print("Successfully read %d pieces" %(cnt))
    
    # Set maximum length
    melody_segs_length = segments_length*16+segments_length-1
    chord_segs_length = segments_length*4+segments_length-1

    # Padding input and one-hot vectorization of output data
    input_melody_left = pad_sequences(input_melody_left, padding='post', maxlen=melody_segs_length)
    input_melody_mid = pad_sequences(input_melody_mid, padding='post', maxlen=16)
    input_melody_right = pad_sequences(input_melody_right, padding='post', maxlen=melody_segs_length)
    input_chord_left = pad_sequences(input_chord_left, padding='post', maxlen=chord_segs_length)
    output_chord = to_categorical(pad_sequences(output_chord, padding='post', maxlen=4), num_classes=16).transpose((1,0,2))
    
    if len(input_melody_left_val)!=0:

        input_melody_left_val = pad_sequences(input_melody_left_val, padding='post', maxlen=melody_segs_length)
        input_melody_mid_val = pad_sequences(input_melody_mid_val, padding='post', maxlen=16)
        input_melody_right_val = pad_sequences(input_melody_right_val, padding='post', maxlen=melody_segs_length)
        input_chord_left_val = pad_sequences(input_chord_left_val, padding='post', maxlen=chord_segs_length)
        output_chord_val = to_categorical(pad_sequences(output_chord_val, padding='post', maxlen=4), num_classes=16).transpose((1,0,2))
    
    return (input_melody_left, input_melody_mid, input_melody_right, input_chord_left, output_chord), \
           (input_melody_left_val, input_melody_mid_val, input_melody_right_val, input_chord_left_val, output_chord_val)


def build_chord_model(segments_length, rnn_size, num_layers, weights_path=None):

    # Set maximum length
    melody_segs_length = segments_length*16+segments_length-1
    chord_segs_length = segments_length*4+segments_length-1

    # Create input layer with embedding
    input_melody_left = Input(shape=(melody_segs_length,), 
                              name='input_melody_left')
    embeded_melody_left = Embedding(input_dim=132,
                                    output_dim=12,
                                    input_length=melody_segs_length,
                                    mask_zero=True,
                                    name='embeded_melody_left')(input_melody_left)

    input_melody_mid = Input(shape=(16,), 
                             name='input_melody_mid')
    embeded_melody_mid = Embedding(input_dim=132,
                                   output_dim=12,
                                   input_length=16,
                                   mask_zero=True,
                                   name='embeded_melody_mid')(input_melody_mid)

    input_melody_right = Input(shape=(melody_segs_length,), 
                                      name='input_melody_right')
    embeded_melody_right = Embedding(input_dim=132,
                                     output_dim=12,
                                     input_length=melody_segs_length,
                                     mask_zero=True,
                                     name='embeded_melody_right')(input_melody_right)

    input_chord_left = Input(shape=(chord_segs_length,), 
                                    name='input_chord_left')
    embeded_chord_left = Embedding(input_dim=15,
                                   output_dim=4,
                                   input_length=chord_segs_length,
                                   mask_zero=True,
                                   name='embeded_chord_left')(input_chord_left)
    
    # Update variable names
    melody_left = embeded_melody_left
    melody_mid = embeded_melody_mid
    melody_right = embeded_melody_right
    chord_left = embeded_chord_left

    return_sequences = True

    # Creating the hidden layer of the LSTM
    for idx in range(num_layers):

        if idx == num_layers - 1:

            return_sequences = False

        melody_left = LSTM(units=rnn_size, 
                           return_sequences=return_sequences,
                           name='melody_left_'+str(idx+1))(melody_left)

        melody_mid = LSTM(units=rnn_size,
                          return_sequences=return_sequences,
                          name='melody_mid_'+str(idx+1))(melody_mid)

        melody_right = LSTM(units=rnn_size, 
                            return_sequences=return_sequences,
                            name='melody_right_'+str(idx+1))(melody_right)

        chord_left = LSTM(units=rnn_size, 
                          return_sequences=return_sequences,
                          name='chord_left_'+str(idx+1))(chord_left)

    # Merge hidden layer output
    merge = concatenate(
                        [
                         melody_left,
                         melody_mid,
                         melody_right,
                         chord_left
                        ]
                       )        

    # Linear transformation of the merged vector
    merge= Dense(units=rnn_size,
                 activation='relu',
                 name='merge')(merge)
    predictions = BatchNormalization()(merge)

    # Create output layer
    first = Dense(units=16, 
                  activation="softmax", 
                  name='first')(predictions)
    second = Dense(units=16, 
                   activation="softmax", 
                   name='second')(predictions)
    third = Dense(units=16, 
                  activation="softmax", 
                  name='third')(predictions)
    fourth = Dense(units=16, 
                   activation="softmax", 
                   name='fourth')(predictions)

    model = Model(
                  inputs=[
                          input_melody_left,
                          input_melody_mid,
                          input_melody_right,
                          input_chord_left
                          ],
                  outputs=[
                           first,
                           second,
                           third,
                           fourth
                          ]
                 )

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    if weights_path==None:

        model.summary()

    else:

        model.load_weights(weights_path)

    return model


def train_model(data, 
                data_val, 
                segments_length=SEGMENTS_LENGTH, 
                rnn_size=CHO_RNN_SIZE, 
                num_layers=CHO_NUM_LAYERS, 
                batch_size=CHO_BATCH_SIZE, 
                epochs=CHO_EPOCHS, 
                verbose=2,
                weights_path='chord_'+WEIGHTS_PATH):

    model = build_chord_model(segments_length, rnn_size, num_layers)

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
                                'input_chord_left': np.array(data[3])},
                            y={'first': np.array(data[4][0]), 
                                'second': np.array(data[4][1]), 
                                'third': np.array(data[4][2]), 
                                'fourth': np.array(data[4][3])},
                            validation_data=({'input_melody_left': np.array(data_val[0]), 
                                                'input_melody_mid': np.array(data_val[1]), 
                                                'input_melody_right': np.array(data_val[2]), 
                                                'input_chord_left': np.array(data_val[3])}, 
                                                {'first': np.array(data_val[4][0]), 
                                                'second': np.array(data_val[4][1]), 
                                                'third': np.array(data_val[4][2]), 
                                                'fourth': np.array(data_val[4][3])}),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[checkpoint])
    
    else:
        
        # Without validation set
        history = model.fit(x={'input_melody_left': np.array(data[0]), 
                                'input_melody_mid': np.array(data[1]), 
                                'input_melody_right': np.array(data[2]), 
                                'input_chord_left': np.array(data[3])},
                            y={'first': np.array(data[4][0]), 
                                'second': np.array(data[4][1]), 
                                'third': np.array(data[4][2]), 
                                'fourth': np.array(data[4][3])},
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