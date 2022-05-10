# Path setting
DATASET_PATH = "dataset"
CORPUS_PATH = "data_corpus.bin"
CHORD_TYPES_PATH = 'chord_types.bin'
WEIGHTS_PATH = 'weights.hdf5'
INPUTS_PATH = "inputs"
OUTPUTS_PATH = "outputs"

# 'loader.py'
EXTENSION = ['.musicxml', '.xml', '.mxl']

# '.model.py'
VAL_RATIO = 0.1
DROPOUT = 0.2
SEGMENT_LENGTH = 32
RNN_SIZE = 128
NUM_LAYERS = 3
BATCH_SIZE = 2048
EPOCHS = 20

# 'harmonizor.py'
RHYTHM_DENSITY = 0.6
CHORD_PER_BAR = False
REPEAT_CHORD = False
WATER_MARK = True