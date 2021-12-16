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
  
The main differences with the cleaned version of Jukedeck are as follows:  
1.　Save these files as lead sheets (.mxl) instead of .mid;  
2.　Corrected mistakes of the cleaned version of Jekedeck about 5% of tunes;  
3.　Added the corresponding title for each piece of music.  
  
Click here to download [Nottingham Lead Sheet Dataset](https://drive.google.com/file/d/1j7MOdTwEASn7wQ7M46NfBUc-ztY5T17K/view?usp=sharing) (Google Drive).  
  
<div align="center">
  <img src=https://github.com/sander-wood/autoharmonizer/blob/homepage/readme/Picture2-1.png width=35% />
  <img src=https://github.com/sander-wood/autoharmonizer/blob/homepage/readme/Picture1-1.png width=35% />
    
  Ashover27 sheet exported in MuseScore3 (left: Jukedeck, right: Nottingham Lead Sheet Dataset)
</div>
  
## Session Lead Sheet Dataset
We create a lead sheet dataset based on [Session Dataset](https://thesession.org/), named as Session Lead Sheet Dataset, containing 40,925 tunes with chords. This dataset is collected as follows.  
1.　We first downloaded all the tunes in ABC format from the Session Dataset, a community website dedicated to Irish traditional music;  
2.　We then convert those ABC files to MusicXML with the [music21 toolkit](https://web.mit.edu/music21/doc/moduleReference/index.html);  
3.　We cleaned the converted files and removed the repeat notation by flattening each score to make them more machine-readable;  
4.　We use AutoHarmonizer to generate the corresponding harmonies for these Irish traditional tunes.  
  
Each harmonized piece contains melody and corresponding chord progression, and metadata information such as key signature, time signature, title and its genre.  
  
<div align="center">
  <img src=https://github.com/sander-wood/autoharmonizer/blob/homepage/readme/Picture4-1.png width=35% />
  <img src=https://github.com/sander-wood/autoharmonizer/blob/homepage/readme/Picture3-1.png width=35% />
    
  Barndance104 sheet exported in Notepad and MuseScore3 (left: Session Dataset, right: Session Lead Sheet Dataset)
</div>
  
Session Lead Sheet Dataset can be used but not limited to the following research topics including: 1) harmonic study, 2) ethnomusicological study, 3) melody harmonization and 4) melody generation based on chords.  
  
Although the chords are machine-generated, the AutoHarmonizer is closer to human-composed chord progressions than other melody harmonization systems, as it takes into account harmonic rhythms.  
  
In addition, given that Ireland and Britain share a very similar cultural background, using the AutoHarmonizer trained on Nottingham Lead Sheet Dataset to produce the chord progressions for the Session Dataset would be more in keeping with its melodic style.  
  
We suggest using this dataset for pre-training and later fine-tune on a dataset like Nottingham Lead Sheet Dataset to further improve the performance of deep learning models.
  
Click here to download [Session Lead Sheet Dataset](https://drive.google.com/file/d/1jGDzip0ODImbgMThrqj8_wXVXu2WaAi8/view?usp=sharing) (Google Drive).  
  
|  Dataset   | Notes  | Chords  | Bars  | Pieces  |
| :----: | :----: | :----: | :----: | :----: |
| Fiddle Tunes | 48,321 | 4,978 | 8,128 | 226 |
| Nottingham | 189,215 | 51,342 | 38,821 | 1,034 |
| RJ Tunebook | 193,916 | 24,930 | 38,218 | 1,078 |
| Wikifonia | 932,813 | 330,241 | 496,437 | 6,244 |
| TheoryTab | 869,052 | 284,936 | 180,488 | 18,167 |
| Session | **7,783,509** | **1,638,386** | **1,353,370** | **40,925** |
  
## Install Dependencies
Python: 3.7.9  
Keras: 2.3.0  
keras-metrics: 1.1.0  
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
