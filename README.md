# AutoHarmonizer

## Generating Chords from Melody with Flexible Harmonic Rhythm and Controllable Harmonic Density

This is the source code of AutoHarmonizer, a harmonic density-controllable melody harmonization system with flexible harmonic rhythm, trained/validated on Wikifonia.org's lead sheet dataset.  
  
The input melodies and harmonized samples are in the `inputs` and `outputs` folders respectively.  
  
The musical discrimination test is available at https://sander-wood.github.io/autoharmonizer/test.  
  
For more information, see our paper: [arXiv paper](https://arxiv.org/abs/2112.11122).
  
## Install Dependencies
Python: 3.7.9  
Keras: 2.3.0  
keras-metrics: 1.1.0  
tensorflow-gpu: 2.2.0  
music21: 6.7.1  
tqdm: 4.62.3  
samplings: 0.1.7
  
PS: Third party libraries can be installed using the `pip install` command.

## Melody Harmonization
1.　Put the melodies (could be parsed by [music21](https://web.mit.edu/music21/doc/moduleReference/moduleConverter.html?highlight=converter#module-music21.converter)) in the `inputs` folder;  
2.　Simply run `harmonizer.py`;  
3.　Wait a while and the harmonized melodies will be saved in the `outputs` folder.  
  
You can set the parameter RHYTHM_DENSITY∈[0, 1] in `config.py` to adjust the density of the generated chord progression. The higher the value of RHYTHM_DENSITY, the more chords will be generated, and vice versa.  

## Use Your Own Dataset
1.　Store all the lead sheets (MusicXML) in the `dataset` folder;  
2.　Run `loader.py`, which will generate `data_corpus.bin`;  
3.　Run `model.py`, which will generate `weights.hdf5`.  
  
After that, you can use `harmonizer.py` to harmonize music with chord progressions that fit the musical style of the new dataset.   
  
If you need to finetune the parameters, you can do so in `config.py`. It is not recommended to change the parameters in other files.
