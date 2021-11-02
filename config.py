# 路径设定
DATASET_PATH = "dataset"
CORPUS_PATH = "corpus.bin"
WEIGHTS_PATH = 'weights.hdf5'
INPUTS_PATH = "inputs"
OUTPUTS_PATH = "outputs"

# 'loader.py'参数
EXTENSION = ['.musicxml', '.xml', '.mxl', '.midi', '.mid', '.krn']

# 'rhythm_model.py'和'chord_model.py'参数
VAL_RATIO = 0
SEGMENT_LENGTH = 32
SEGMENTS_LENGTH = 2
RNN_SIZE = 256
NUM_LAYERS = 1
BATCH_SIZE = 32
RHYTHM_EPOCHS = 3
CHORD_EPOCHS = 4

# 'harmonizor.py'参数
TEMPERATURE = 0
RHYTHM_DENSITY = 0