# Path setting
DATASET_PATH = "dataset"
CORPUS_PATH = "corpus.bin"
WEIGHTS_PATH = 'weights.hdf5'
INPUTS_PATH = "inputs"
OUTPUTS_PATH = "outputs"

# 'loader.py'
EXTENSION = ['.musicxml', '.xml', '.mxl']

# 'harmonic_rhythm_model.py'
HAR_VAL_RATIO = 0.1
SEGMENT_LENGTH = 32
HAR_RNN_SIZE = 128
HAR_NUM_LAYERS = 2
HAR_BATCH_SIZE = 64
HAR_EPOCHS = 4

# 'chord_model.py'
CHO_VAL_RATIO = 0.1
SEGMENTS_LENGTH = 2
CHO_RNN_SIZE = 128
CHO_NUM_LAYERS = 3
CHO_BATCH_SIZE = 64
CHO_EPOCHS = 5

# 'harmonizer.py'
RHYTHM_DENSITY = 0.5
WATER_MARK = True
LEADSHEET = True
