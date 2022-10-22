---
layout: default
title: Data Representation
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
        }
    });
</script>

# Data Representation

Our data representation takes into account two pieces of meta-information from sheet music, namely the time signature and the key signature. As shown in Fig. 1, each lead sheet is encoded into four sequences of the same length.

<br>
<center><img src="figs/fig1.png" alt="fig1" style="zoom:80%"></center>
<br>
Figure 1: A two-bar sample of a melody, beat, key, and chord representation (at a time resolution of eighth notes).
<br>

**Melody Sequence**: we use 128-dimensional one-hot vector to represent each frame, with a time resolution of sixteenth notes. The first dimension represents rests, while the remaining dimensions correspond to the 127 different pitches in MIDI (except 0).

**Beat Sequence**: a sequence of 4-dimensional vectors based on time signatures. It represents the beat strength of frames in the melody sequence. Its values range from 0 to 3, corresponding to four categories: non-beat, weak, medium-weight, and strong beat.

**Key Sequence**: the keys are encoded according to their number of sharps, with flats corresponding to $-7$ to $-1$, no flat/sharp corresponding to 0, and sharps corresponding to 1 to 7, for a total of 15 types.

**Chord Sequence**: each chord is encoded as a one-hot vector: the first dimension represents rests, while the others correspond to 1,461 chord symbols.
