import harmonic_rhythm_model
import chord_model
from config import *

# Train harmonic rhythm model
data, data_val = harmonic_rhythm_model.create_training_data()
history = harmonic_rhythm_model.train_model(data, data_val, weights_path='rhythm_'+WEIGHTS_PATH)

# Train chord model
data, data_val = chord_model.create_training_data()
history = chord_model.train_model(data, data_val, weights_path='chord_'+WEIGHTS_PATH)