import numpy as np
import keras_metrics as km
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
from loader import *

def mul_seg(sequence, idx, segments_length, direction, sep):

    segs = []

    # 判定连接方向
    if direction=='left':

        for i in range(-segments_length,0):

            segs += sequence[idx+i]

            # 添加'分隔符'
            if i!=-1:

                segs.append(sep)
    
    elif direction=='right':

        for i in range(1,segments_length+1):

            segs += sequence[idx+i]

            # 添加'分隔符'
            if i!=segments_length:

                segs.append(sep)

    return segs
        
        
def create_training_data(segments_length=SEGMENTS_LENGTH, corpus_path=CORPUS_PATH,  val_ratio=VAL_RATIO):

    # 加载语料库
    with open('chord_'+corpus_path, "rb") as filepath:
        corpus = pickle.load(filepath)

    # 训练集的输入和输出
    input_melody_left = []
    input_melody_mid = []
    input_melody_right = []
    input_chord_left = []
    output_chord = []

    # 验证集的输入和输出
    input_melody_left_val = []
    input_melody_mid_val = []
    input_melody_right_val = []
    input_chord_left_val = []
    output_chord_val = []

    cnt = 0

    # 处理语料库中的每一个和弦序列
    for idx, chord in enumerate(corpus[1]):
        
        # 加载对应的旋律片段序列
        melody = corpus[0][idx]
        
        # 以'131','13'分别表示旋律以及和弦序列的填充符
        for i in range(segments_length):

            melody.insert(0,[131]*16)
            melody.append([131]*16)
            chord.insert(0,[13]*4)

        # 创建数据对
        for i in range(segments_length, len(chord)):
            
            # 以'132'和'14'分别表示旋律以及和弦序列的分隔符
            melody_left = mul_seg(melody,i,segments_length,'left',132)
            melody_mid = melody[i]
            melody_right = mul_seg(melody,i,segments_length,'right',132)[::-1]
            chord_left = mul_seg(chord,i,segments_length,'left',14)
            target = chord[i]

            # 按一定概率随机分配到训练集或验证集
            if np.random.rand()>val_ratio:

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
    
    # 定义最大长度
    melody_segs_length = segments_length*16+segments_length-1
    chord_segs_length = segments_length*4+segments_length-1

    # 补齐输入并独热化输出数据
    input_melody_left = pad_sequences(input_melody_left, padding='post', maxlen=melody_segs_length)
    input_melody_mid = pad_sequences(input_melody_mid, padding='post', maxlen=16)
    input_melody_right = pad_sequences(input_melody_right, padding='post', maxlen=melody_segs_length)
    input_chord_left = pad_sequences(input_chord_left, padding='post', maxlen=chord_segs_length)
    output_chord = to_categorical(pad_sequences(output_chord, padding='post', maxlen=4), num_classes=15).transpose((1,0,2))
    
    if len(input_melody_left_val)!=0:

        input_melody_left_val = pad_sequences(input_melody_left_val, padding='post', maxlen=melody_segs_length)
        input_melody_mid_val = pad_sequences(input_melody_mid_val, padding='post', maxlen=16)
        input_melody_right_val = pad_sequences(input_melody_right_val, padding='post', maxlen=melody_segs_length)
        input_chord_left_val = pad_sequences(input_chord_left_val, padding='post', maxlen=chord_segs_length)
        output_chord_val = to_categorical(pad_sequences(output_chord_val, padding='post', maxlen=4), num_classes=15).transpose((1,0,2))
    
    return (input_melody_left, input_melody_mid, input_melody_right, input_chord_left, output_chord), \
           (input_melody_left_val, input_melody_mid_val, input_melody_right_val, input_chord_left_val, output_chord_val)


def build_chord_model(weights_path=None):

    # 定义最大长度
    melody_segs_length = SEGMENTS_LENGTH*16+SEGMENTS_LENGTH-1
    chord_segs_length = SEGMENTS_LENGTH*4+SEGMENTS_LENGTH-1

    # 创建输入层
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
    
    # 创建隐藏层 
    melody_left = LSTM(units=RNN_SIZE, 
                    name='melody_left')(embeded_melody_left)
    melody_mid = LSTM(units=RNN_SIZE,
                   name='melody_mid')(embeded_melody_mid)
    melody_right = LSTM(units=RNN_SIZE, 
                        name='melody_right')(embeded_melody_right)
    chord_left = LSTM(units=RNN_SIZE, 
                      name='chord_left')(embeded_chord_left)

    # 合并隐藏层输出
    merge = concatenate(
        [
            melody_left,
            melody_mid,
            melody_right,
            chord_left
        ]
    )        

    # 对合并向量进行线性变换
    merge= Dense(units=RNN_SIZE,
                activation='relu',
                name='merge')(merge)
    predictions = BatchNormalization()(merge)

    # 创建输出层
    first = Dense(units=15, 
                 activation="softmax", 
                 name='first')(predictions)
    second = Dense(units=15, 
                   activation="softmax", 
                   name='second')(predictions)
    third = Dense(units=15, 
                  activation="softmax", 
                  name='third')(predictions)
    fourth = Dense(units=15, 
                   activation="softmax", 
                   name='fourth')(predictions)

    model = Model(
            inputs=[input_melody_left,
                    input_melody_mid,
                    input_melody_right,
                    input_chord_left
                    ],
            outputs=[first,
                     second,
                     third,
                     fourth
                    ]
        )

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', km.f1_score()])

    if weights_path==None:

        model.summary()

    else:

        model.load_weights(weights_path)

    return model


def train_model(data, data_val, weights_path='chord_'+WEIGHTS_PATH):

    # 创建和声模型
    model = build_chord_model()

    # 加载或删除已有权重
    if os.path.exists(weights_path):
        
        try:

            model.load_weights(weights_path)
            print("checkpoint loaded")
        
        except:

            os.remove(weights_path)
            print("checkpoint deleted")

    # 设定监控指标
    if len(data_val[0])!=0:

        monitor = 'val_loss'

    else:

        monitor = 'loss'
        
    # 保存权重
    checkpoint = ModelCheckpoint(filepath=weights_path,
                                 monitor=monitor,
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')

    # 训练模型
    if len(data_val[0])!=0:

        # 加入验证集
        model.fit(x={'input_melody_left': np.array(data[0]), 
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
                batch_size=BATCH_SIZE,
                epochs=CHORD_EPOCHS,
                callbacks=[checkpoint])
    
    else:
        
        # 仅包括训练集
        model.fit(x={'input_melody_left': np.array(data[0]), 
                    'input_melody_mid': np.array(data[1]), 
                    'input_melody_right': np.array(data[2]), 
                    'input_chord_left': np.array(data[3])},
                y={'first': np.array(data[4][0]), 
                    'second': np.array(data[4][1]), 
                    'third': np.array(data[4][2]), 
                    'fourth': np.array(data[4][3])},
                batch_size=BATCH_SIZE,
                epochs=CHORD_EPOCHS,
                callbacks=[checkpoint])


if __name__ == "__main__":

    # 加载训练集和验证集
    data, data_val = create_training_data()
    
    # 训练模型
    train_model(data, data_val)