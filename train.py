import rhythm_model
import chord_model

# 加载训练集和验证集
data, data_val = rhythm_model.create_training_data()

# 训练模型
history = rhythm_model.train_model(data, data_val)

# 加载训练集和验证集
data, data_val = chord_model.create_training_data()

# 训练模型
history = chord_model.train_model(data, data_val)