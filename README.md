# AutoHarmonizer

## Melody Harmonization with Controllable Harmonic Rhythm

This is the source code of AutoHarmonizer, a harmonic rhythm-controllable melody harmonization system. \
\
The input melodies and harmonized samples are in the `inputs` and `outputs` folders respectively.\
\
Musical discrimination test is available at https://sander-wood.github.io/autoharmonizer. \
\
Nottingham Lead Sheet Dataset (NLSD) and Session Lead Sheet Dataset (SLSD) are both in the `dataset` folder.\
\
For more information, see our paper: [arXiv paper](https://www.overleaf.com/project/61837c3a1936bf9bea54a14a).

## Install Dependencies
Python: 3.7.9\
Keras: 2.3.0\
tensorflow-gpu: 2.2.0\
music21: 6.7.1\
\
PS: Third party libraries can be installed using the `pip install` command.

## Melody Harmonization
1.　Put the melodies (MIDI or MusicXML) in the `inputs` folder;\
2.　Simply run `harmonizer.py`;\
3.　Wait a while and the harmonized melodies will be saved in the `outputs` folder.\
\
PS: You can set the parameter RHYTHM_DENSITY∈(0, 1) in `config.py` to adjust the density of the generated chord progression. The lower the value of RHYTHM_DENSITY, the fewer chords will be generated, and vice versa.

## Use Your Own Dataset
1.　Store all the lead sheets (MIDI or MusicXML) in the `dataset` folder;\
2.　Run `loader.py`, which will generate `rhythm_corpus.bin` and `chord_corpus.bin`; \
3.　Run `train.py`, which will generate `rhythm_weights.hdf5` and `chord_weights.hdf5`.\
\
After that, you can use `harmonizer.py` to harmonize music that with chord progressions that fit the musical style of the new dataset. \
\
If you need to finetune the parameters, you can do so in `config.py`. It is not recommended to change the parameters in other files.
