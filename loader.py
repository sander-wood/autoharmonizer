import os
import pickle
from config import *
from music21 import *
from tqdm import trange

def part(score):

    try:

        score = score.parts[0]

    except:

        score = score
    
    return score


def norm_pos(pos):

    # Calculate extra position
    extra_pos = pos%0.25

    # If greater than 0
    if extra_pos>0:

        pos = pos-extra_pos+0.25
    
    return pos


def norm_duration(element):

    # Read the duration
    note_duration = element.quarterLength
    
    # Calculate positions of note
    note_start = element.offset
    note_end = note_start + note_duration

    # Regularized position and duration
    note_start = norm_pos(note_start)
    note_end = norm_pos(note_end)
    note_duration = note_end-note_start

    return note_duration


def transpose(score):

    # Set default interval, key signature and tempo
    gap = interval.Interval(0)
    ks = key.KeySignature(0)
    tp = tempo.MetronomeMark(number=120)

    for element in score.flat:
        
        # Found key signature
        if isinstance(element, key.KeySignature) or isinstance(element, key.Key):

            if isinstance(element, key.KeySignature):

                ks = element.asKey()
            
            else:

                ks = element

            # Identify the tonic
            if ks.mode == 'major':
                
                tonic = ks.tonic

            else:

                tonic = ks.parallel.tonic

            # Transpose score
            gap = interval.Interval(tonic, pitch.Pitch('C'))
            score = score.transpose(gap)

            break
        
        # Found tempo
        elif isinstance(element, tempo.MetronomeMark):

            tp = element

        # No key signature found
        elif isinstance(element, note.Note) or \
             isinstance(element, note.Rest) or \
             isinstance(element, chord.Chord):
            
            break

        else:

            continue

    return score, gap, ks, tp


def beat_seq(ts):

    # Read time signature
    beatCount = ts.numerator
    beatDuration = 4/ts.denominator

    # Create beat sequence
    beat_sequence = [0]*beatCount*int(beatDuration/0.25)
    beat_sequence[0] += 1

    
    # Check if the numerator is divisible by 3 or 2
    medium = 0 

    if (ts.numerator%3)==0:

        medium = 3

    elif (ts.numerator%2)==0:

        medium = 2

    for idx in range(len(beat_sequence)):

        # Add 1 to each beat
        if idx%((beatDuration/0.25))==0:

            beat_sequence[idx] += 1
        
        # Mark medium-weight beat (at every second or third beat)
        if (medium==3 and idx%((3*beatDuration/0.25))==0) or \
            (medium==2 and idx%((2*beatDuration/0.25))==0):

            beat_sequence[idx] += 1
            
    return beat_sequence


def chord2vec(element):

    if isinstance(element, chord.Chord):

        # Extracts the MIDI pitch of each note in a chord
        pitch_list = [sub_ele.pitch.midi for sub_ele in element.notes]
        pitch_list = sorted(pitch_list)
    
    elif isinstance(element, note.Rest):

        # Four '13' to indicate that it is a 'rest'
        return [13]*4

    # Reduced MIDI pitch range
    first_note = pitch_list[0]
    pitch_list = [num-first_note for num in pitch_list]
    pitch_list = [first_note%12]+pitch_list[1:]

    vec = []

    # All notes within one octave (range 1 to 12)
    for i, element in enumerate(pitch_list):

        if element<12 and i<4:

            vec.append(element+1)

    # Padding
    vec = vec + [0]*(4-len(vec))

    return vec


def leadsheet_converter(score):

    # Initialization
    melody_part = []
    chord_part = [] 
    chord_list = []
    
    # Read lead sheet
    for element in part(score).flat:
        
        # If is ChordSymbol
        if isinstance(element, harmony.ChordSymbol):
            
            chord_list.append(element)

        else:

            melody_part.append(element)

    # If no chord at the beginning
    if chord_list[0].offset!=0:

        first_rest = note.Rest()
        first_rest.quarterLength = chord_list[0].offset
        chord_part.append(first_rest)

    # Instantiated chords
    for idx in range(1, len(chord_list)):

        new_chord = chord.Chord(chord_list[idx-1].notes)
        new_chord.offset = chord_list[idx-1].offset
        new_chord.quarterLength = chord_list[idx].offset-chord_list[idx-1].offset
        chord_part.append(new_chord)
    
    # Add the last chord
    new_chord = chord.Chord(chord_list[-1].notes)
    new_chord.offset = chord_list[-1].offset
    new_chord.quarterLength = melody_part[-1].offset-chord_list[idx].offset
    chord_part.append(new_chord)

    return stream.Part(melody_part).flat, stream.Part(chord_part).flat


def chord2txt(chord_part):

    # Initialization
    rhythm_txt = []
    chord_segs = []

    # Read chord list from chord part
    chord_list = []

    for element in chord_part.flat:

        if isinstance(element, chord.Chord) or isinstance(element, note.Rest):

            # Read the regularized duration
            element.quarterLength = norm_duration(element)
            
            # Skip if the duration is equal to 0 after regularization
            if element.quarterLength==0:

                continue

            # Correct abnormal duration
            if len(chord_list)!=0 and element.quarterLength<=(chord_list[-1].quarterLength/8):
                
                corrected_duration = element.quarterLength+chord_list[-1].quarterLength
                chord_list[-1].quarterLength = corrected_duration
                continue
            
            else:

                chord_list.append(element)
    
    # Read chord list
    for idx, element in enumerate(chord_list):

        # Read vectorized chord
        chord_segs.append(chord2vec(element))

        if isinstance(element, note.Rest):

            # '0' for rests
            rhythm_txt.append(0)
            
        else:

            # '1' for chord onset
            rhythm_txt.append(1)
        
        # Read the duration of chord
        chord_duration = element.quarterLength
        
        # '2' for harmonic rhythm holding
        rhythm_txt += [2]*int(chord_duration/0.25-1)
        
    return rhythm_txt, chord_segs


def melody2txt(melody_part):

    # Initialization
    pre_ele = None
    melody_txt = []
    beat_txt = []
    ts_seq = []
    
    # Read note and meta information from melody part
    for element in melody_part.flat:
        
        if isinstance(element, note.Note) or isinstance(element, note.Rest):
            
            # Read the regularized duration
            note_duration = norm_duration(element)
            
            # Skip if the duration is equal to 0 after regularization
            if note_duration==0:

                continue

            # Reads the MIDI pitch of a note (value range 1 to 128)
            if isinstance(element, note.Note):

                melody_txt.append(element.pitch.midi+1)
            
            # '129' for rest
            elif isinstance(element, note.Rest):

                # Merge adjacent rests
                if isinstance(pre_ele, note.Rest):

                    melody_txt.append(130)
                
                else:

                    melody_txt.append(129)
            
            # '130' for holding
            note_steps = int(note_duration/0.25)
            melody_txt += [130]*(note_steps-1)

            # Save current note
            pre_ele = element

        # Read the current time signature
        elif isinstance(element, meter.TimeSignature):

            ts_seq.append(element)
    
    # Initialization
    cur_cnt = 0
    pre_cnt = 0
    beat_sequence = beat_seq(meter.TimeSignature('c'))

    # create beat sequence
    if len(ts_seq)!=0:

        # Traverse time signartue sequence
        for ts in ts_seq:
            
            # Calculate current time step
            cur_cnt = ts.offset/0.25

            if cur_cnt!=0:
                
                # Fill in the previous beat sequence
                beat_txt += beat_sequence*int((cur_cnt-pre_cnt)/len(beat_sequence))

                # Complete the beat sequence
                missed_beat = int((cur_cnt-pre_cnt)%len(beat_sequence))

                if missed_beat!=0:

                    beat_txt += beat_sequence[:missed_beat]

            # Update variables
            beat_sequence = beat_seq(ts)
            pre_cnt = cur_cnt

    # Handle the last time signature
    cur_cnt = len(melody_txt)
    beat_txt += beat_sequence*int((cur_cnt-pre_cnt)/len(beat_sequence))

    # Complete the beat sequence
    missed_beat = int((cur_cnt-pre_cnt)%len(beat_sequence))

    if missed_beat!=0:

        beat_txt += beat_sequence[:missed_beat]

    return melody_txt, beat_txt, ts_seq


def music2txt(score, filename, fromDataset):

    # Read data from dataset
    if fromDataset:
        
        # Transpose to C-major/A-minor
        score, gap, ks, tp = transpose(score)

        try:

            # Read melody and chord part
            melody_part, chord_part = leadsheet_converter(score)

        except:

            # Read error
            print("Warning: Failed to convert \"%s\"" %filename)
            return None

        # Read melody and chord data
        melody_txt, beat_txt, ts_seq = melody2txt(melody_part)
        rhythm_txt, chord_segs = chord2txt(chord_part)
        
        if melody_txt!=None:
            
            if len(rhythm_txt)!=len(melody_txt):

                upper = min(len(rhythm_txt),len(melody_txt))

                # Guaranteed equal length of melody, beat and rhythm sequence
                melody_txt = melody_txt[:upper]
                beat_txt = beat_txt[:upper]
                rhythm_txt = rhythm_txt[:upper]

            # Create melody segment
            melody_seg = []
            melody_segs = []

            for idx, token in enumerate(rhythm_txt):

                if idx==(len(rhythm_txt)-1) or (token!=2 and len(melody_seg)!=0):
                    
                    # Add the last one
                    if idx==(len(rhythm_txt)-1):

                        melody_seg.append(melody_txt[idx])

                    # Over-length truncation
                    if len(melody_seg)>16:
                        
                        # Make sure melody_seg is within a whole note
                        melody_seg = melody_seg[:16]
                        
                    melody_segs.append(melody_seg)
                    melody_seg = []
                    melody_seg.append(melody_txt[idx])

                else:

                    melody_seg.append(melody_txt[idx])
            
            if len(melody_segs)!=len(chord_segs):

                upper = min(len(melody_segs),len(chord_segs))

                # Guaranteed equal length of melody and chord segment sequence
                melody_segs = melody_segs[:upper]
                chord_segs = chord_segs[:upper]

    # Read melody part
    else:

        # Transpose to C-major/A-minor
        melody_part, gap, ks, tp = transpose(score)

        # Read melody, beat and time signature sequences
        melody_txt, beat_txt, ts_seq = melody2txt(melody_part)

        return melody_txt, beat_txt, gap, ks, tp, ts_seq

    return melody_txt, beat_txt, rhythm_txt, melody_segs, chord_segs
        

def key_split(score):

    scores = []
    score_part = []
    ks = None
    ts = None
    pre_offset = 0

    for element in part(score).flat:

        # If is key signature
        if isinstance(element, key.KeySignature) or isinstance(element, key.Key):

            # If is not the first key signature
            if ks!=None:

                scores.append(stream.Stream(score_part))
                ks = element
                pre_offset = ks.offset
                ks.offset = 0
                new_ts = meter.TimeSignature(ts.ratioString)
                score_part = [ks, new_ts]
            
            else:

                ks = element
                score_part.append(ks)

        # If is time signature
        elif isinstance(element, meter.TimeSignature):

            element.offset -= pre_offset
            ts = element
            score_part.append(element)
        
        else:

            element.offset -= pre_offset
            score_part.append(element)

    scores.append(stream.Stream(score_part))

    return scores


def music_loader(path=DATASET_PATH, fromDataset=True):

    # Initialization
    melody_data = []
    beat_data = []
    rhythm_data = []
    melody_segs_data = []
    chord_segs_data = []
    gap_data = []
    meta_data = []
    melody_parts = []
    filenames = []

    # Traverse the path
    for dirpath, dirlist, filelist in os.walk(path):
        
        # Traverse the list of files
        for file_idx in trange(len(filelist)):

            this_file = filelist[file_idx]

            # Ensure that suffixes in the training set are valid
            if fromDataset and os.path.splitext(this_file)[-1] not in EXTENSION:

                continue
        
            filename = os.path.join(dirpath, this_file)

            try:
                
                # Read the this music file
                score = converter.parse(filename)
                scores = key_split(score)

                if fromDataset:

                    for score in scores:

                        # Converte music to text data
                        melody_txt, beat_txt, rhythm_txt, melody_segs, chord_segs = music2txt(score, filename, fromDataset=True)
                        
                        if melody_txt!=None:
                            
                            melody_data.append(melody_txt)
                            beat_data.append(beat_txt)
                            rhythm_data.append(rhythm_txt)
                            melody_segs_data.append(melody_segs)
                            chord_segs_data.append(chord_segs)
                    
                else:

                    melody_txts = []
                    beat_txts = []
                    gaps = []
                    metas = []

                    for score in scores:

                        # Converte music to text data
                        melody_txt, beat_txt, gap, ks, tp, ts_seq = music2txt(score, filename, fromDataset=False)

                        if melody_txt!=None:

                            melody_txts.append(melody_txt)
                            beat_txts.append(beat_txt)
                            gaps.append(gap)
                            metas.append([ks, tp, ts_seq])

                    melody_data.append(melody_txts)
                    beat_data.append(beat_txts)
                    gap_data.append(gaps)
                    meta_data.append(metas)
                    melody_parts.append(scores)
                    filenames.append(this_file)
                    
            except:

                # Unable to read this music file
                print("Warning: Failed to read \"%s\"" %filename)
                continue
            
    print("Successfully encoded %d pieces" %(len(melody_data)))  
    
    if fromDataset:

        return (melody_data, beat_data, rhythm_data, melody_segs_data, chord_segs_data)
    
    else:

        return (melody_data, beat_data, gap_data, meta_data, melody_parts, filenames)


if __name__ == "__main__":

    # Read encoded music information and file names
    corpus = music_loader()
    
    # Save as corpus
    with open('rhythm_'+CORPUS_PATH, "wb") as filepath:
        pickle.dump(corpus[:3], filepath)

    with open('chord_'+CORPUS_PATH, "wb") as filepath:
        pickle.dump(corpus[3:], filepath)