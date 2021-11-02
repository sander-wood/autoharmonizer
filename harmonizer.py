import numpy as np
from music21 import note
from music21 import chord
from music21 import stream
from loader import music_loader
from rhythm_model import build_rhythm_model
from chord_model import build_chord_model
from chord_model import mul_seg
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from config import *

def sample(prediction, last_note=False, temperature=TEMPERATURE, rhythm_density=RHYTHM_DENSITY):
    
    # 判断prediction是否一维 (chord)
    if prediction.shape[0]==1:

        prediction = prediction.tolist()[0]

        # 置零不可能的值
        prediction[14] = 0
        prediction[15] = 0

        if not last_note:

            prediction[0] = 0
    
    else:
        
        # 记录非保持符的概率
        rest_prob = 1-prediction[2]

        # 提升其余字符概率
        for idx in range(len(prediction)):

            if idx!=2:

                prediction[idx] += prediction[2]*rhythm_density*(prediction[idx]/rest_prob)

        # 抑制保持符概率
        prediction[2] = prediction[2]*(1-rhythm_density)

    # 判断温度参数是否为0
    if temperature==0:

        # 确定性采样
        return np.argmax(prediction)
    
    else:

        # 改变概率向量的概率分布
        prediction = np.log(prediction) / temperature
        probabilites = np.exp(prediction) / np.sum(np.exp(prediction))

        # 随机采样
        index = np.random.choice(range(len(probabilites)), p=probabilites)

        return index


def generate_rhythm(rhythm_model, melody_data, beat_data, segment_length=SEGMENT_LENGTH):

    rhythm_data = []

    # 处理语料库中的每一个旋律序列
    for idx, melody in enumerate(melody_data):
        
        # 加载对应的节拍序列
        beat = beat_data[idx]
        
        # 以'131', '4', '3'分别表示旋律，节拍和节奏序列的填充符
        melody = [131]*segment_length + melody_data[idx] + [131]*segment_length
        beat = [4]*segment_length + beat + [4]*segment_length
        rhythm = [3]*segment_length

        # 逐个预测
        for i in range(segment_length, len(melody)-segment_length):

            # 创建输入数据
            input_melody_left = melody[i-segment_length: i] 
            input_melody_mid = melody[i]
            input_melody_right = melody[i+1: i+segment_length+1][::-1]
            input_beat_left = beat[i-segment_length: i] 
            input_beat_mid = beat[i]
            input_beat_right = beat[i+1: i+segment_length+1][::-1]
            input_rhythm_left = rhythm[i-segment_length: i]
            
            # 独热化输入数据
            input_melody_left = to_categorical(input_melody_left, num_classes=132)[np.newaxis, ...]
            input_melody_mid = to_categorical(input_melody_mid, num_classes=132)[np.newaxis, ...]
            input_melody_right = to_categorical(input_melody_right, num_classes=132)[np.newaxis, ...]
            input_beat_left = to_categorical(input_beat_left, num_classes=5)[np.newaxis, ...]
            input_beat_mid = to_categorical(input_beat_mid, num_classes=5)[np.newaxis, ...]
            input_beat_right = to_categorical(input_beat_right, num_classes=5)[np.newaxis, ...]
            input_rhythm_left = to_categorical(input_rhythm_left, num_classes=4)[np.newaxis, ...]

            # 预测下一个节奏
            prediction = rhythm_model.predict(x=[input_melody_left, input_melody_mid, input_melody_right, input_beat_left, input_beat_mid, input_beat_right, input_rhythm_left])[0]
            rhythm_idx = sample(prediction)

            # 更新输入序列
            rhythm.append(rhythm_idx)
        
        # 去除前面的'填充符'
        rhythm_data.append(rhythm[segment_length:])
    
    return rhythm_data


def generate_chord(chord_model, melody_data, rhythm_data, segments_length=SEGMENTS_LENGTH):

    chord_data = []

    # 处理语料库中的每一个旋律序列
    for idx, melody in enumerate(melody_data):

        # 加载对应的节奏序列
        rhythm = rhythm_data[idx]
        melody_segs = []
        chord_segs = []

        # 以'131','14'分别表示旋律以及和弦序列的填充符
        for i in range(segments_length):

            if i!=segments_length-1:

                melody_segs.append([131]*16)

            chord_segs.append([14]*4)
            
        melody_seg = [131]*16
    
        # 读取节奏序列中的每一个token
        for t_idx, token in enumerate(rhythm):
            
            # 判断是否为'保持符' 
            if token<2:
                
                # 判断是否为'休止符'
                if token==0:

                    chord_segs.append([13]*4)

                else:

                    chord_segs.append(['chord'])

                # 确保melody_seg在四拍内
                if len(melody_seg)>16:

                    melody_seg = melody_seg[:16]
                    
                melody_segs.append(melody_seg)
                melody_seg = [melody[t_idx]]

                # 若读到最后一位
                if (t_idx+1)==len(rhythm):
                    
                    # 确保melody_seg在四拍内
                    if len(melody_seg)>16:

                        melody_seg = melody_seg[:16]
                    
                    melody_segs.append(melody_seg)  

            else:

                melody_seg.append(melody[t_idx])     
                
                # 若读到最后一位
                if (t_idx+1)==len(rhythm):
                    
                    # 确保melody_seg在四拍内
                    if len(melody_seg)>16:

                        melody_seg = melody_seg[:16]
                    
                    melody_segs.append(melody_seg)  
                    
        # 以'131'表示旋律序列的填充符
        for i in range(segments_length):

            melody_segs.append([131]*16)

        # 定义最大长度
        melody_segs_length = segments_length*16+segments_length-1
        chord_segs_length = segments_length*4+segments_length-1
        
        for i in range(segments_length, len(chord_segs)):
            
            # 若当前的是休止符则跳过
            if len(chord_segs[i])==[13, 13, 13, 13]:

                continue

            # 以'132'和'15'分别表示旋律以及和弦序列的分隔符
            input_melody_left = mul_seg(melody_segs,i,segments_length,'left',132)
            input_melody_mid = melody_segs[i]
            input_melody_right = mul_seg(melody_segs,i,segments_length,'right',132)[::-1]
            input_chord_left = mul_seg(chord_segs,i,segments_length,'left',15)
            
            # 补齐输入数据
            input_melody_left = pad_sequences([input_melody_left], padding='post', maxlen=melody_segs_length)
            input_melody_mid = pad_sequences([input_melody_mid], padding='post', maxlen=16)
            input_melody_right = pad_sequences([input_melody_right], padding='post', maxlen=melody_segs_length)
            input_chord_left = pad_sequences([input_chord_left], padding='post', maxlen=chord_segs_length)

            # 预测下一个和弦
            predictions = chord_model.predict(x=[input_melody_left, input_melody_mid, input_melody_right, input_chord_left]) 
            
            # 采样
            first = sample(predictions[0])
            second = sample(predictions[1])
            third = sample(predictions[2])
            fourth = sample(predictions[3], last_note=True)

            # 更新输入序列
            chord_segs[i] = [first, second, third, fourth]
        
        # 去除开头的填充符
        chord_segs = chord_segs[segments_length:]
        
        cnt = 0
        chord = []
        
        # 重构和声信息
        for t_idx, token in enumerate(rhythm):

            # 读到'和弦'
            if token<2:
                
                cur_chord = chord_segs[cnt]

                if cur_chord==[13,13,13,13]:

                    chord.append(129)
                    cnt += 1
                    continue
                
                # 去除和声列表'0'
                for cur_idx in range(4):

                    if cur_chord[cur_idx]==0:

                        del cur_chord[cur_idx]
                    
                    else:

                        cur_chord[cur_idx]-=1

                bias = cur_chord[0]+48
                cur_chord = [bias]+[cur_token+bias for cur_token in cur_chord[1:]]
                chord.append(cur_chord)
                cnt += 1
            
            # 读到'保持符'
            else:

                chord.append(130)

        chord_data.append(chord)
    
    return chord_data


def txt2music(txt, gap, meta):

    # 初始化
    notes = [meta[0], meta[1]]
    pre_element = None
    duration = 0.0
    offset = 0.0
    corrected_gap = -1*(gap.semitones)

    # 解码文本序列
    for element in txt+[131]:
        
        if element!=130:

            # 创建新音符
            if pre_element!=None:

                if isinstance(pre_element, int):

                    # 若发现音符
                    if pre_element<129:

                        new_note = note.Note(element-1+corrected_gap)

                    # 若发现休止符
                    elif pre_element==129:

                        new_note = note.Rest()
                    
                # 若发现和弦
                else:

                    new_note = chord.Chord([note.Note(cur_note+corrected_gap) for cur_note in pre_element])
                
                new_note.quarterLength = duration
                new_note.offset = offset
                notes.append(new_note)
            
            # 更新offset和时值并保存当前音符
            offset += duration
            duration = 0.25
            pre_element = element
            
            # 检查是否更新拍号
            if len(meta[2])!=0:

                if meta[2][0].offset<=offset:

                    # 更新拍号
                    notes.append(meta[2][0])
                    del meta[2][0]
        
        else:
            
            # 更新时值
            duration += 0.25

    return stream.Part(notes)


def export_midi(melody_parts, chord_data, gap_data, meta_data, filenames, output_path='outputs'):

    # 遍历所有旋律
    for idx, melody_part in enumerate(melody_parts):

        # 和声序列转换为和声声部
        chord = txt2music(chord_data[idx], gap_data[idx], meta_data[idx])

        # 保存为midi
        score = stream.Stream([melody_part, chord])
        score.write('mid', fp=output_path+'/'+filenames[idx].split('.')[-2]+'.mid')


if __name__ == "__main__":

    # 从输入文件夹读取数据
    melody_data, beat_data, gap_data, meta_data, melody_parts, filenames = music_loader(path='inputs', fromDataset=False)

    # 构建节奏模型并生成节奏信息
    rhythm_model = build_rhythm_model(SEGMENT_LENGTH, RNN_SIZE, NUM_LAYERS, 'rhythm_'+WEIGHTS_PATH)
    rhythm_data = generate_rhythm(rhythm_model, melody_data, beat_data)
    
    # 构建和声模型并生成和声信息
    chord_model = build_chord_model(SEGMENTS_LENGTH, RNN_SIZE, NUM_LAYERS, 'chord_'+WEIGHTS_PATH)
    chord_data = generate_chord(chord_model, melody_data, rhythm_data)
    
    # 输出midi音乐文件
    export_midi(melody_parts, chord_data, gap_data, meta_data, filenames)