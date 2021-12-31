import numpy as np
import math
from music21 import *
from loader import music_loader
from harmonic_rhythm_model import build_rhythm_model
from chord_model import build_chord_model
from chord_model import mul_seg
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from tqdm import trange
from config import *

def sample(prediction, last_note=False, rhythm_density=RHYTHM_DENSITY):
    
    # If the prediction is from the chord model
    if prediction.shape[0]==1:

        prediction = prediction.tolist()[0]

        # Zeroing impossible values
        prediction[14] = 0
        prediction[15] = 0

        if not last_note:

            prediction[0] = 0
    
    else:
        
        # Set upper and lower bounds for rhythm density
        rhythm_density = max(0.01, rhythm_density)
        rhythm_density = min(0.99, rhythm_density)

        # Save the probabilities of non-holding tokens
        rest_prob = 1-prediction[2]

        # Change the holding token's probability
        old_prob = prediction[2]
        prediction[2] = prediction[2]**math.tan(math.pi*rhythm_density/2)

        # Change the non-holding tokens' probabilities
        for idx in range(len(prediction)):

            if idx!=2:

                prediction[idx] += (old_prob-prediction[2])*(prediction[idx]/rest_prob)

    # Greedy sampling
    return np.argmax(prediction)
    

def generate_rhythm(rhythm_model, melody_data, beat_data, segment_length=SEGMENT_LENGTH):

    rhythm_data = []

    # Process each melody sequence in the corpus
    for idx, melody in enumerate(melody_data):
        
        # Load the corresponding beat sequence
        beat = beat_data[idx]
        
        # '131', '4', '3' for melody, beat and rhythm sequences of paddings respectively
        melody = [131]*segment_length + melody_data[idx] + [131]*segment_length
        beat = [4]*segment_length + beat + [4]*segment_length
        rhythm = [3]*segment_length

        # Predict each token
        for i in range(segment_length, len(melody)-segment_length):

            # Create input data
            input_melody_left = melody[i-segment_length: i] 
            input_melody_mid = melody[i]
            input_melody_right = melody[i+1: i+segment_length+1][::-1]
            input_beat_left = beat[i-segment_length: i] 
            input_beat_mid = beat[i]
            input_beat_right = beat[i+1: i+segment_length+1][::-1]
            input_rhythm_left = rhythm[i-segment_length: i]
            
            # One-hot vectorization
            input_melody_left = to_categorical(input_melody_left, num_classes=132)[np.newaxis, ...]
            input_melody_mid = to_categorical(input_melody_mid, num_classes=132)[np.newaxis, ...]
            input_melody_right = to_categorical(input_melody_right, num_classes=132)[np.newaxis, ...]
            input_beat_left = to_categorical(input_beat_left, num_classes=5)[np.newaxis, ...]
            input_beat_mid = to_categorical(input_beat_mid, num_classes=5)[np.newaxis, ...]
            input_beat_right = to_categorical(input_beat_right, num_classes=5)[np.newaxis, ...]
            input_rhythm_left = to_categorical(input_rhythm_left, num_classes=4)[np.newaxis, ...]

            # Predict the next rhythm
            prediction = rhythm_model.predict(x=[input_melody_left, input_melody_mid, input_melody_right, input_beat_left, input_beat_mid, input_beat_right, input_rhythm_left])[0]
            rhythm_idx = sample(prediction)

            # Updata rhythm sequence
            rhythm.append(rhythm_idx)
        
        # Remove the leading padding 
        rhythm_data.append(rhythm[segment_length:])
    
    return rhythm_data


def generate_chord(chord_model, melody_data, rhythm_data, segments_length=SEGMENTS_LENGTH):

    chord_data = []

    # Process each melody sequence in the corpus
    for idx, melody in enumerate(melody_data):

        # Load the corresponding rhythm sequence
        rhythm = rhythm_data[idx]
        melody_segs = []
        chord_segs = []

        # '131', '14' for melody and chord sequences of paddings respectively
        for i in range(segments_length):

            if i!=segments_length-1:

                melody_segs.append([131]*16)

            chord_segs.append([14]*4)
            
        melody_seg = [131]*16
    
        # Read tokens from rhythm sequence
        for t_idx, token in enumerate(rhythm):
            
            # If is non-holding token
            if token<2:
                
                # If is rest
                if token==0:

                    chord_segs.append([13]*4)

                else:

                    chord_segs.append(['chord'])

                # Make sure melody_seg is within a whole note
                if len(melody_seg)>16:

                    melody_seg = melody_seg[:16]
                    
                melody_segs.append(melody_seg)
                melody_seg = [melody[t_idx]]

                if (t_idx+1)==len(rhythm):
                    
                    # Make sure melody_seg is within a whole note
                    if len(melody_seg)>16:

                        melody_seg = melody_seg[:16]
                    
                    melody_segs.append(melody_seg)  

            else:

                melody_seg.append(melody[t_idx])     
                
                if (t_idx+1)==len(rhythm):
                    
                    # Make sure melody_seg is within a whole note
                    if len(melody_seg)>16:

                        melody_seg = melody_seg[:16]
                    
                    melody_segs.append(melody_seg)  
                    
        # '131' for melody sequence of padding
        for i in range(segments_length):

            melody_segs.append([131]*16)

        # Set maximum length
        melody_segs_length = segments_length*16+segments_length-1
        chord_segs_length = segments_length*4+segments_length-1
        
        for i in range(segments_length, len(chord_segs)):
            
            # Skip if the current one is a rest
            if len(chord_segs[i])==[13, 13, 13, 13]:

                continue

            # '131', '15' for melody and chord sequences of separators respectively
            input_melody_left = mul_seg(melody_segs,i,segments_length,'left',132)
            input_melody_mid = melody_segs[i]
            input_melody_right = mul_seg(melody_segs,i,segments_length,'right',132)[::-1]
            input_chord_left = mul_seg(chord_segs,i,segments_length,'left',15)
            
            # Padding input
            input_melody_left = pad_sequences([input_melody_left], padding='post', maxlen=melody_segs_length)
            input_melody_mid = pad_sequences([input_melody_mid], padding='post', maxlen=16)
            input_melody_right = pad_sequences([input_melody_right], padding='post', maxlen=melody_segs_length)
            input_chord_left = pad_sequences([input_chord_left], padding='post', maxlen=chord_segs_length)

            # Predict the next chord
            predictions = chord_model.predict(x=[input_melody_left, input_melody_mid, input_melody_right, input_chord_left]) 
            
            first = sample(predictions[0])
            second = sample(predictions[1])
            third = sample(predictions[2])
            fourth = sample(predictions[3], last_note=True)

            # Updata chord sequence
            chord_segs[i] = [first, second, third, fourth]
        
        # Remove the leading padding 
        chord_segs = chord_segs[segments_length:]
        
        cnt = 0
        chord = []
        
        # Create chord data
        for t_idx, token in enumerate(rhythm):

            # If is onset
            if token<2:
                
                cur_chord = chord_segs[cnt]

                if cur_chord==[13,13,13,13]:

                    chord.append(129)
                    cnt += 1
                    continue
                
                # Remove '0'
                for cur_idx in range(4):

                    if cur_chord[cur_idx]==0:

                        del cur_chord[cur_idx]
                    
                    else:

                        cur_chord[cur_idx]-=1

                bias = cur_chord[0]+48
                cur_chord = [bias]+[cur_token+bias for cur_token in cur_chord[1:]]
                chord.append(cur_chord)
                cnt += 1
            
            # If is holding
            else:

                chord.append(130)

        chord_data.append(chord)
    
    return chord_data


def txt2music(txt, gap, meta):

    # Initialization
    notes = [meta[0], meta[1]]
    pre_element = None
    duration = 0.0
    offset = 0.0
    corrected_gap = -1*(gap.semitones)

    # Decode text sequences
    for element in txt+[131]:
        
        if element!=130:

            # Create new note
            if pre_element!=None:

                if isinstance(pre_element, int):

                    # If is note
                    if pre_element<129:

                        new_note = note.Note(pre_element-1+corrected_gap)

                    # If is rest
                    elif pre_element==129:

                        new_note = note.Rest()
                    
                # If is chord
                else:

                    new_note = chord.Chord([note.Note(cur_note+corrected_gap) for cur_note in pre_element])
                
                new_note.quarterLength = duration
                new_note.offset = offset
                notes.append(new_note)
            
            # Updata offset, duration and save the element
            offset += duration
            duration = 0.25
            pre_element = element
            
            # Updata time signature
            if len(meta[2])!=0:

                if meta[2][0].offset<=offset:

                    notes.append(meta[2][0])
                    del meta[2][0]
        
        else:
            
            # Updata duration
            duration += 0.25

    return notes


def score_converter(melody_part, chord_part):
    
    # Initialization
    score = []
    chord_part = [element for element in chord_part if isinstance(element, chord.Chord)]

    # Read melody part
    for element in melody_part:
        
        # If is note and chord offset not greater than note offset
        if (isinstance(element, note.Note) or isinstance(element, note.Rest) or isinstance(element, chord.Chord)) \
            and len(chord_part)>0 and element.offset>=chord_part[0].offset:

            # Converted to ChordSymbol
            try:
                chord_symbol = harmony.chordSymbolFromChord(chord_part[0])
                chord_symbol.offset = chord_part[0].offset

                score.append(chord_symbol)
                del chord_part[0]
            
            except:
                
                # Illegal ChordSymbol, converted to a major triad
                new_chord = [n for n in chord_part[0]]
                chord_symbol = harmony.ChordSymbol(root=new_chord[0].name, kind='major')
                chord_symbol.offset = chord_part[0].offset

                score.append(chord_symbol)
                del chord_part[0]
        
        score.append(element)
    
    # Save as mxl
    score = stream.Stream(score)

    return score


def merge_scores(scores):

    score = []
    extra_offset = 0
    
    # Traverse all music
    for sub_score in scores:

        for element in sub_score:

            element.offset += extra_offset
            score.append(element)
        
        # Add additional bias offset
        extra_offset = element.offset+element.quarterLength
    
    return stream.Stream(score)


def watermark(score, filename, water_mark=WATER_MARK):

    # Add water mark
    if water_mark:
        
        score.metadata = metadata.Metadata()
        score.metadata.title = filename
        score.metadata.composer = 'harmonized by AutoHarmonizer'
    
    return score


def export_music(melody_part, chord_data, gap_data, meta_data, filename, output_path=OUTPUTS_PATH, leadsheet=LEADSHEET):

    chord_part = []

    # Traverse all harmonies
    for idx in range(len(chord_data)):

        # Chord sequence to chord part
        chord_subpart = txt2music(chord_data[idx], gap_data[idx], meta_data[idx])
        chord_subpart = stream.Stream(chord_subpart)
        chord_part.append(chord_subpart)

    melody_part = merge_scores(melody_part)
    chord_part = merge_scores(chord_part)

    if leadsheet:
        
        try:

            # Export as leadsheet
            score = score_converter(melody_part, chord_part)
            score = watermark(score, filename.split('.')[-2])
            score.write('mxl', fp=output_path+'/'+filename.split('.')[-2]+'.mxl')
        
        except:
    
            # Export as midi
            print('Warning: failed to export %s as lead sheet, now exporting as midi...' %(filename))
            score = stream.Stream([melody_part, chord_part])
            score = watermark(score, filename.split('.')[-2])
            score.write('mid', fp=output_path+'/'+filename.split('.')[-2]+'.mid')
            
    else:
    
        # Export as midi
        score = stream.Stream([melody_part, chord_part])
        score = watermark(score, filename.split('.')[-2])
        score.write('mid', fp=output_path+'/'+filename.split('.')[-2]+'.mid')
        
        

if __name__ == "__main__":

    # Load data from 'inputs'
    melody_data, beat_data, gap_data, meta_data, melody_parts, filenames = music_loader(path=INPUTS_PATH, fromDataset=False)

    # Build harmonic rhythm and chord model
    rhythm_model = build_rhythm_model(SEGMENT_LENGTH, HAR_RNN_SIZE, HAR_NUM_LAYERS, 'rhythm_'+WEIGHTS_PATH)
    chord_model = build_chord_model(SEGMENTS_LENGTH, CHO_RNN_SIZE, CHO_NUM_LAYERS, 'chord_'+WEIGHTS_PATH)
    
    # Process each melody sequence
    for idx in trange(len(melody_data)):
        
        # Generate harmonic rhythm and chord data
        rhythm_data = generate_rhythm(rhythm_model, melody_data[idx], beat_data[idx])
        chord_data = generate_chord(chord_model, melody_data[idx], rhythm_data)

        # Export music file
        export_music(melody_parts[idx], chord_data, gap_data[idx], meta_data[idx], filenames[idx])