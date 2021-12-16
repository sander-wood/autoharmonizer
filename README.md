# AutoHarmonizer

## Melody Harmonization with Controllable Harmonic Rhythm

This is the source code of AutoHarmonizer, a harmonic rhythm-controllable melody harmonization system, trained/validated on Nottingham Lead Sheet Dataset.  
  
The input melodies and harmonized samples are in the `inputs` and `outputs` folders respectively.  
  
Musical discrimination test is available at https://sander-wood.github.io/autoharmonizer.  
  
For more information, see our paper: [arXiv paper](https://www.overleaf.com/project/61837c3a1936bf9bea54a14a).  
  
## Nottingham Lead Sheet Dataset

The music generation start-up company Jukedeck put some efforts into cleaning the database and released at https://github.com/jukedeck/nottingham-dataset in MIDI format.  
  
But we still found some mistakes of this version about 5% of tunes (e.g. mismatches, or no harmonies at all).  
  
Therefore, we manually corrected the MIDI version cleaned by Jukedeck, and all the tunes now are titled while present in the form of the lead sheet.  
  
Here is a sample comparison.  
  
<div align="center">
  <img src=https://github.com/sander-wood/autoharmonizer/blob/homepage/readme/Picture2-1.png width=35% />
  <img src=https://github.com/sander-wood/autoharmonizer/blob/homepage/readme/Picture1-1.png width=35% />
    
  Ashover27 in the Jukedeck version    
  Ashover27 in the Nottingham Lead Sheet Dataset
</div>

## Install Dependencies
Python: 3.7.9  
Keras: 2.3.0  
tensorflow-gpu: 2.2.0  
music21: 6.7.1  
  
PS: Third party libraries can be installed using the `pip install` command.

## Melody Harmonization
1.　Put the melodies (MIDI or MusicXML) in the `inputs` folder;  
2.　Simply run `harmonizer.py`;  
3.　Wait a while and the harmonized melodies will be saved in the `outputs` folder.  
  
You can set the parameter RHYTHM_DENSITY∈(0, 1) in `config.py` to adjust the density of the generated chord progression. The lower the value of RHYTHM_DENSITY, the fewer chords will be generated, and vice versa.  

## Use Your Own Dataset
1.　Store all the lead sheets (MIDI or MusicXML) in the `dataset` folder;  
2.　Run `loader.py`, which will generate `rhythm_corpus.bin` and `chord_corpus.bin`;  
3.　Run `train.py`, which will generate `rhythm_weights.hdf5` and `chord_weights.hdf5`.  
  
After that, you can use `harmonizer.py` to harmonize music that with chord progressions that fit the musical style of the new dataset.   
  
If you need to finetune the parameters, you can do so in `config.py`. It is not recommended to change the parameters in other files.
