import os
import pickle
from config import *
from music21 import converter
from music21 import note
from music21 import chord
from music21 import meter
from music21 import key
from music21 import pitch
from music21 import interval
from music21 import tempo
from fractions import Fraction

def norm_pos(pos):

    # 计算位置余量
    extra_pos = pos%0.25

    # 若位置余量大于0
    if extra_pos>0:

        pos = pos-extra_pos+0.25
    
    return pos


def norm_duration(element):

    # 读取音符的时值(特殊时值映射为常规时值，过短时值设为16分音符)
    note_duration = element.quarterLength
    
    # 计算音符的开始和结束位置
    note_start = element.offset
    note_end = note_start + note_duration

    # 正则化音符位置并计算正则后的时值
    note_start = norm_pos(note_start)
    note_end = norm_pos(note_end)
    note_duration = note_end-note_start

    return note_duration


def transpose(score):

    # 设定默认移调距离和调号
    gap = interval.Interval(0)
    ks = key.KeySignature(0)
    tp = tempo.MetronomeMark(number=120)

    for element in score.recurse():
        
        # 发现调号
        if isinstance(element, key.KeySignature) or isinstance(element, key.Key):

            if isinstance(element, key.KeySignature):

                ks = element.asKey()
            
            else:

                ks = element

            # 确认主音
            if ks.mode == 'major':
                
                tonic = ks.tonic

            else:

                tonic = ks.parallel.tonic

            # 对乐谱进行移调
            gap = interval.Interval(tonic, pitch.Pitch('C'))
            score = score.transpose(gap)

            break
        
        # 找到速度标记
        elif isinstance(element, tempo.MetronomeMark):

            tp = element

        # 未找到调号
        elif isinstance(element, note.Note) or \
             isinstance(element, note.Rest) or \
             isinstance(element, chord.Chord):
            
            break
        
        # 无关紧要的对象
        else:

            continue
    
    return score, gap, ks, tp


def beat_seq(ts):

    # 读取拍号信息
    beatCount = ts.numerator
    beatDuration = 4/ts.denominator

    # 生成节拍序列
    beat_sequence = [0]*beatCount*int(beatDuration/0.25)
    beat_sequence[0] += 1

    
    # 检查分子是否可被3或2整除
    second = 0 

    if (ts.numerator%3)==0:

        second = 3

    elif (ts.numerator%2)==0:

        second = 2

    # 遍历所有单元
    for idx in range(len(beat_sequence)):

        # 给每个拍点加1
        if idx%((beatDuration/0.25))==0:

            beat_sequence[idx] += 1
        
        # 标记次强拍 (每二拍或三拍处)
        if (second==3 and idx%((3*beatDuration/0.25))==0) or \
            (second==2 and idx%((2*beatDuration/0.25))==0):

            beat_sequence[idx] += 1
            
    return beat_sequence


def chord2vec(element):

    if isinstance(element, chord.Chord):

        # 提取出和弦内各个音的MIDI音高
        pitch_list = [sub_ele.pitch.midi for sub_ele in element.notes]
        pitch_list = sorted(pitch_list)
    
    elif isinstance(element, note.Rest):

        # 返回4个'13'表示当前和弦为'休止符'
        return [13]*4

    # 缩减MIDI音高值域
    first_note = pitch_list[0]
    pitch_list = [num-first_note for num in pitch_list]
    pitch_list = [first_note%12]+pitch_list[1:]

    vec = []

    # 确保音符数量小于4并且都在一个八度以内 (值域1~12)
    for i, element in enumerate(pitch_list):

        if element<12 and i<4:

            vec.append(element+1)

    # 补零对齐
    vec = vec + [0]*(4-len(vec))

    return vec


def melody2txt(melody_part, filename):

    # 初始化
    pre_ele = None
    pre_duration = 0
    melody_txt = []
    beat_txt = []
    ts_cnt = 0
    ts_seq = []
    
    # 从旋律声部读取音符及元信息
    for element in melody_part.recurse():
        
        if isinstance(element, note.Note) or isinstance(element, note.Rest) or isinstance(element, chord.Chord):
            
            # 若原始时值等于'0'则退出报错
            if element.quarterLength==0:

                melody_txt = None
                print("Warning: Found a duration equal to 0 in \"%s\"" %filename)
                break
            
            # 读取正则化时值
            note_duration = norm_duration(element)
            
            # 若正则化后时值等于'0'则跳过
            if note_duration==0:

                continue

            # 读取音符的MIDI音高（值域1~128）
            if isinstance(element, note.Note):

                melody_txt.append(element.pitch.midi+1)
            
            # 用‘129’表示'休止符'的MIDI音高
            elif isinstance(element, note.Rest):

                # 合并连续'休止符'或防止出现异常'休止符'
                if (isinstance(element, note.Rest) and isinstance(pre_ele, note.Rest)) or (len(melody_txt)!=0 and note_duration<=(pre_duration/7)):
                    
                    # 计算并存储修正的时值
                    missed_duration = [130]*int(note_duration/0.25)
                    melody_txt += missed_duration
                    pre_ele = element
                    continue

                else:

                    melody_txt.append(129)

            # 读取和弦最高音作为当前音符
            else:

                note_list = [sub_ele.pitch.midi for sub_ele in element.notes]
                note_list = sorted(note_list)
                melody_txt.append(note_list[-1])
            
            # 以'130'表示单个长为16分音符的'保持符'
            note_steps = int(note_duration/0.25)
            melody_txt += [130]*(note_steps-1)

            # 读取音符的offset
            note_offset = Fraction(element.offset)
            
            # 若offset未前进则退出报错
            if note_offset<pre_duration:
                
                melody_txt = None
                print("Warning: Found multiple voices with in one track in \"%s\"" %filename)
                break

            else:

                pre_duration += Fraction(element.quarterLength)

            pre_ele = element

        # 读取当前拍号
        elif isinstance(element, meter.TimeSignature):

            ts_seq.append(element)
    
    # 初始化
    cur_cnt = 0
    pre_cnt = 0
    beat_sequence = beat_seq(meter.TimeSignature('c'))

    # 创建节拍序列
    if len(ts_seq)!=0:

        # 逐个读取拍号
        for ts in ts_seq:
            
            # 计算当前时间步
            cur_cnt = ts.offset/0.25

            # 若当前时间步非零
            if cur_cnt!=0:
                
                # 补上先前的节拍序列
                beat_txt += beat_sequence*int((cur_cnt-pre_cnt)/len(beat_sequence))

                # 计算不完全的节拍序列
                missed_beat = (cur_cnt-pre_cnt)%len(beat_sequence)

                if missed_beat!=0:

                    beat_txt += beat_sequence[:missed_beat]

            # 更新变量
            beat_sequence = beat_seq(ts)
            pre_cnt = cur_cnt

    # 处理最后一个拍号
    cur_cnt = len(melody_txt)
    beat_txt += beat_sequence*int((cur_cnt-pre_cnt)/len(beat_sequence))

    # 计算不完全的节拍序列
    missed_beat = int((cur_cnt-pre_cnt)%len(beat_sequence))

    if missed_beat!=0:

        beat_txt += beat_sequence[:missed_beat]

    return melody_txt, beat_txt, ts_cnt, beat_sequence, ts_seq


def music2txt(score, filename, fromDataset):

    # 初始化
    rhythm_txt = []
    melody_segs = []
    chord_segs = []

    # 从dataset里读数据
    if fromDataset:
        
        # 转调至C大调/A小调
        score, gap, ks, tp = transpose(score)

        try:

            # 读取旋律声部以及和声声部
            melody_part = score.parts[0].flat
            chord_part = score.parts[1].flat
        
        except:

            print("Warning: Failed to read \"%s\"" %filename)
            return None

        # 从和声声部读取得到和弦列表
        chord_list = []

        for element in chord_part.recurse():

            if isinstance(element, chord.Chord) or isinstance(element, note.Rest):

                # 防止出现异常时值
                if len(chord_list)!=0 and element.quarterLength<=(chord_list[-1].quarterLength/7):
                    
                    # 计算并存储修正的时值
                    corrected_duration = element.quarterLength+chord_list[-1].quarterLength
                    chord_list[-1].quarterLength = corrected_duration
                    continue

                # 合并连续休止符时值
                elif isinstance(element, note.Rest) and len(chord_list)!=0 and isinstance(chord_list[-1], note.Rest):

                    chord_list[-1].quarterLength += element.quarterLength
                
                else:

                    chord_list.append(element)
        
        # 读取和弦列表
        for idx, element in enumerate(chord_list):

            # 读取当前向量化的和弦
            chord_segs.append(chord2vec(element))

            if isinstance(element, note.Rest):

                # 用'0'表示'休止符'的Onset
                rhythm_txt.append(0)
                
            else:

                # 用'1'表示'和弦'的Onset
                rhythm_txt.append(1)
            
            # 防止出现异常时值
            chord_duration = element.quarterLength

            if idx>0:

                last_chord_duration = chord_list[idx-1].quarterLength

                if abs(chord_duration-last_chord_duration)<(last_chord_duration/7):

                    chord_duration = last_chord_duration
            
            # 以'2'表示单个长为16分音符的'保持符'
            rhythm_txt += [2]*int(chord_duration/0.25-1)

        # 读取旋律序列, 节拍序列等信息
        melody_txt, beat_txt, ts_cnt, beat_sequence, ts_seq = melody2txt(melody_part, filename)
        
        if melody_txt!=None:
            
            if len(rhythm_txt)!=len(melody_txt):

                upper = min(len(rhythm_txt),len(melody_txt))

                # 保证旋律和节拍以及节奏序列等长
                melody_txt = melody_txt[:upper]
                beat_txt = beat_txt[:upper]
                rhythm_txt = rhythm_txt[:upper]

            else:

                # 补齐三个序列保证节拍按照拍号填满
                if (len(melody_txt)>ts_cnt):

                    missed_num = (len(melody_txt)-ts_cnt)%len(beat_sequence)
                
                    if missed_num>0:

                        melody_txt += (len(beat_sequence)-missed_num)*[130]
                        rhythm_txt += (len(beat_sequence)-missed_num)*[2]
                        
                if len(melody_txt)>len(beat_txt):

                    beat_txt += beat_sequence[len(beat_txt)-len(melody_txt):]

            # 创建旋律片段
            melody_seg = []

            for idx, token in enumerate(rhythm_txt):

                if idx==(len(rhythm_txt)-1) or (token!=2 and len(melody_seg)!=0):
                    
                    # 补上最后一位
                    if idx==(len(rhythm_txt)-1):

                        melody_seg.append(melody_txt[idx])

                    # 过长截断
                    if len(melody_seg)>16:
                        
                        # 确保melody_seg在四拍内
                        melody_seg = melody_seg[:16]
                        
                    melody_segs.append(melody_seg)
                    melody_seg = []
                    melody_seg.append(melody_txt[idx])

                else:

                    melody_seg.append(melody_txt[idx])
            
            if len(melody_segs)!=len(chord_segs):

                upper = min(len(melody_segs),len(chord_segs))

                # 保证旋律片段序列以及和声片段序列等长
                melody_segs = melody_segs[:upper]
                chord_segs = chord_segs[:upper]

    # 读取旋律声部
    else:

        # 转调至C大调/A小调
        original_score = score
        score, gap, ks, tp = transpose(score)

        try:

            # 读取旋律声部以及和声声部
            melody_part = score.parts[0].flat
        
        except:
            
            print("Warning: Failed to read \"%s\"" %filename)
            return None

        # 读取旋律序列, 节拍序列等信息
        melody_txt, beat_txt, ts_cnt, beat_sequence, ts_seq = melody2txt(melody_part, filename)

        if melody_txt!=None:

            # 补齐三个序列保证节拍按照拍号填满
                if (len(melody_txt)>ts_cnt):

                    missed_num = (len(melody_txt)-ts_cnt)%len(beat_sequence)
                
                    if missed_num>0:

                        melody_txt += (len(beat_sequence)-missed_num)*[130]
                        
                if len(melody_txt)>len(beat_txt):

                    beat_txt += beat_sequence[len(beat_txt)-len(melody_txt):]

        return melody_txt, beat_txt, gap, ks, tp, ts_seq, original_score.parts[0].flat

    return melody_txt, beat_txt, rhythm_txt, melody_segs, chord_segs
        

def music_loader(path=DATASET_PATH, fromDataset=True):

    # 编码音乐及其文件名
    melody_data = []
    beat_data = []
    rhythm_data = []
    melody_segs_data = []
    chord_segs_data = []
    gap_data = []
    meta_data = []
    melody_parts = []
    filenames = []

    # 遍历文件夹下的所有音乐
    for dirpath, dirlist, filelist in os.walk(path):
        
        # 逐个处理文件
        for this_file in filelist:

            # 确保后缀合法
            if os.path.splitext(this_file)[-1] not in EXTENSION:

                continue
        
            filename = os.path.join(dirpath, this_file)

            try:
                
                # 读取当前音乐
                print("Parsing \"%s\"" %filename)
                score = converter.parse(filename)

                if fromDataset:

                    # 读入数据集下的音乐转换为文本数据
                    melody_txt, beat_txt, rhythm_txt, melody_segs, chord_segs = music2txt(score, filename, fromDataset=True)

                    if melody_txt!=None:
                        
                        melody_data.append(melody_txt)
                        beat_data.append(beat_txt)
                        rhythm_data.append(rhythm_txt)
                        melody_segs_data.append(melody_segs)
                        chord_segs_data.append(chord_segs)
                    
                else:

                    # 读入输入下的音乐转换为文本数据
                    melody_txt, beat_txt, gap, ks, tp, ts_seq, melody_part = music2txt(score, filename, fromDataset=False)

                    if melody_txt!=None:

                        melody_data.append(melody_txt)
                        beat_data.append(beat_txt)
                        gap_data.append(gap)
                        meta_data.append([ks, tp, ts_seq])
                        melody_parts.append(melody_part)
                        filenames.append(this_file)
                    
            except:

                # 无法读取当前音乐
                print("Warning: Failed to read \"%s\"" %filename)
                continue
            
    print("Successfully encoded %d pieces" %(len(melody_data)))  
    
    if fromDataset:

        return (melody_data, beat_data, rhythm_data, melody_segs_data, chord_segs_data)
    
    else:

        return (melody_data, beat_data, gap_data, meta_data, melody_parts, filenames)


if __name__ == "__main__":

    # 读取编码后的音乐信息以及文件名
    corpus = music_loader()
    
    # 保存为语料库
    with open('rhythm_'+CORPUS_PATH, "wb") as filepath:
        pickle.dump(corpus[:3], filepath)

    with open('chord_'+CORPUS_PATH, "wb") as filepath:
        pickle.dump(corpus[3:], filepath)