import os
import rhythm_model
import chord_model

# 根据下面的范围改一个固定数值
segment_length_list = [16, 32, 64]
segments_length_list = [1, 2, 3]

segment_length = 16
segments_length = 1

# Grid Search范围
rnn_size_list = [128, 256, 512]
num_layers_list = [1, 2, 3]
batch_size_list = [32, 64, 128]


min_val_loss = 1000
best_para = []
best_history = None

# 加载训练集和验证集
data, data_val = rhythm_model.create_training_data(segment_length=segment_length)

for rnn_size in rnn_size_list:

    for num_layers in num_layers_list:

        for batch_size in batch_size_list:

            try:

                os.remove('rhythm_weights.hdf5')

            except:

                print('There is nothing to delete.')

            print('Testing rhythm model with rnn_size==%d, num_layers==%d, batch_size==%d' %(rnn_size, num_layers, batch_size))

            # 训练模型
            history = rhythm_model.train_model(data, data_val, segment_length, rnn_size, num_layers, batch_size, verbose=2)

            for epoch, val_loss in enumerate(history.history['val_loss']):

                print('Epoch %d: val_loss=%f.' %(epoch, val_loss))
                
                if val_loss<min_val_loss:

                    min_val_loss = val_loss
                    best_para = [rnn_size, num_layers, batch_size, epoch]
                    best_history = history

            print('Current minimal val_loss %f' %(min_val_loss))

print(best_para)

for key in best_history.keys():

    print(key)
    print(best_history[key][best_para[-1]])

    
min_val_loss = 1000
best_para = []
best_history = None

# 加载训练集和验证集
data, data_val = chord_model.create_training_data(segments_length=segments_length)

for rnn_size in rnn_size_list:

    for num_layers in num_layers_list:

        for batch_size in batch_size_list:

            try:

                os.remove('chord_weights.hdf5')

            except:

                print('There is nothing to delete.')

            print('Testing chord model with rnn_size==%d, num_layers==%d, batch_size==%d' %(rnn_size, num_layers, batch_size))

            # 训练模型
            history = chord_model.train_model(data, data_val, segments_length, rnn_size, num_layers, batch_size, verbose=2)

            for epoch, val_loss in enumerate(history.history['val_loss']):

                print('Epoch %d: val_loss=%f.' %(epoch, val_loss))
                
                if val_loss<min_val_loss:

                    min_val_loss = val_loss
                    best_para = [rnn_size, num_layers, batch_size, epoch]
                    best_history = history

            print('Current minimal val_loss %f' %(min_val_loss))

print(best_para)

for key in best_history.keys():

    print(key)
    print(best_history[key][best_para[-1]])