---
layout: default
title: Dataset
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

# Chordified JSB  Chorales  Dataset 
Since the original JSB Chorales Dataset has no chord progressions and the workload of carrying out harmonic analysis manually is too large, we perform the following automated pre-processing to add chord symbols.  
  
1.　**Flattening**: all repeat barlines are removed by flattening each score to make them more machine-readable.  
2.　**Chordify**: a tool in [music21](https://web.mit.edu/music21/doc/usersGuide/usersGuide_09_chordify.html?highlight=chordify) for simplifying a complex score with multiple parts into a succession of chords in one part.  
3.　**Labelling**: we first move all the chords to the closed position, and then label the chordified chords as chord symbols. Finally, all chord symbols on strong beats of the soprano part are kept.  

After removing a few scores that cannot be properly chordified, we ended up with a total of 366 chorales for training (90\%) and validation (10\%).  

You can find this chordified version of JSB Chorales dataset in the `dataset` folder. 

<div align="center">
  <img src=https://github.com/sander-wood/deepchoir/blob/homepage/figs/070-1.png width=35% />
  <img src=https://github.com/sander-wood/deepchoir/blob/homepage/figs/070-2.png width=35% />
    
  Chordified BWV 322 exported in MuseScore3
</div>