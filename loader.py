import os
import pickle
import numpy as np
from copy import deepcopy
from tqdm import trange
from music21 import *
from config import *

def quant_score(score):
    
    for element in score.flat:
        onset = np.ceil(element.offset/0.25)*0.25

        if isinstance(element, note.Note) or isinstance(element, note.Rest) or isinstance(element, chord.Chord):
            offset = np.ceil((element.offset+element.quarterLength)/0.25)*0.25
            element.quarterLength = offset - onset

        element.offset = onset

    return score


def get_filenames(input_dir):
    
    filenames = []

    # Traverse the path
    for dirpath, dirlist, filelist in os.walk(input_dir):
        # Traverse the list of files
        for this_file in filelist:
            # Ensure that suffixes in the training set are valid
            if input_dir==DATASET_PATH and os.path.splitext(this_file)[-1] not in EXTENSION:
                continue
            filename = os.path.join(dirpath, this_file)
            filenames.append(filename)
    
    return filenames


def melody_reader(score):

    melody_txt = []
    beat_txt = []
    chord_txt = []
    key_txt = []
    sharps = 0
    chord_token = 'R'

    for element in score.flat:

        if isinstance(element, note.Note):
            # midi pitch as note onset
            token = element.pitch.midi

        elif isinstance(element, note.Rest):
            # 0 as rest onset
            token = 0
            
        elif isinstance(element, chord.Chord) and not isinstance(element, harmony.ChordSymbol):
            notes = [n.pitch.midi for n in element.notes]
            notes.sort()
            token = notes[-1]
            
        elif isinstance(element, harmony.ChordSymbol):
            chord_token = element.figure
            continue
        
        elif isinstance(element, key.Key) or isinstance(element, key.KeySignature):
            sharps = element.sharps+8
            continue
            
        else:
            continue
        
        melody_txt += [token]*int(element.quarterLength*4)
        beat_txt += [int(element.beatStrength*4)]*int(element.quarterLength*4)
        key_txt += [sharps]*int(element.quarterLength*4)
        chord_txt += [chord_token]*int(element.quarterLength*4)

    return melody_txt, beat_txt, key_txt, chord_txt


def convert_files(filenames, fromDataset=True):

    print('\nConverting %d files...' %(len(filenames)))
    failed_list = []
    data_corpus = []

    for filename_idx in trange(len(filenames)):

        # Read this music file
        filename = filenames[filename_idx]
        
        try:
            
            score = converter.parse(filename)
            score = score.parts[0]
            if not fromDataset:
                original_score = deepcopy(score)
            song_data = []
            melody_data = []
            beat_data = []
            key_data = []

            score = quant_score(score)
            melody_txt, beat_txt, key_txt, chord_txt = melody_reader(score)

            if fromDataset:
                if len(melody_txt)==len(beat_txt) and len(beat_txt)==len(key_txt) and len(key_txt)==len(chord_txt):
                    song_data.append((melody_txt, beat_txt, key_txt, chord_txt))
                
                else:
                    failed_list.append((filename, 'length mismatch'))
                    song_data = []
                    break

            else:
                if len(melody_txt)!=len(beat_txt) or len(melody_txt)!=len(key_txt):
                    min_len = min(len(melody_txt), len(beat_txt))
                    melody_txt = melody_txt[:min_len]
                    beat_txt = beat_txt[:min_len]
                    key_txt = key_txt[:min_len]
                    
                melody_data.append(melody_txt)
                beat_data.append(beat_txt)
                key_data.append(key_txt)
            
            if not fromDataset:
                data_corpus.append((melody_data, beat_data, key_data, original_score, filename))
            
            elif len(song_data)>0:
                data_corpus.append(song_data)

        except Exception as e:
            failed_list.append((filename, e))

    print('Successfully converted %d files.' %(len(filenames)-len(failed_list)))
    if len(failed_list)>0:
        print('Failed numbers: '+str(len(failed_list)))
        print('Failed to process: \n')
        for failed_file in failed_list:
            print(failed_file)

    if fromDataset:
        chord_types = [song[3] for songs in data_corpus for song in songs]
        chord_types = [item for sublist in chord_types for item in sublist]
        chord_types = list(set(chord_types))
        chord_types.remove('R')
        chord_types = ['R']+chord_types

        with open(CHORD_TYPES_PATH, "wb") as filepath:
            pickle.dump(chord_types, filepath)

        with open(CORPUS_PATH, "wb") as filepath:
            pickle.dump(data_corpus, filepath)
    
    else:
        return data_corpus


if __name__ == '__main__':

    filenames = get_filenames(input_dir=DATASET_PATH)
    convert_files(filenames)