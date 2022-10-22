---
layout: default
title: Home
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

## Generating Chords from Melody with Flexible Harmonic Rhythm and Controllable Harmonic Density

Melody harmonization, i.e., generating a chord progression for a user-given melody, remains a challenging task to this day. A chord progression must not only be in harmony with the melody, but its harmonic rhythm is also interdependent on the melodic rhythm. Although previous neural network-based systems can effectively generate a chord progression for a melody, few studies have addressed controllable melody harmonization, and there has been a lack of focus on generating flexible harmonic rhythms. In this paper, we propose AutoHarmonizer, a harmonic density-controllable melody harmonization system with flexible harmonic rhythm. This system supports 1,462 chord types and can generate denser or sparser chord progressions for a given melody. Experimental results demonstrate the diversity of harmonic rhythms in the AutoHarmonizer-generated chord progressions and the effectiveness of controllable harmonic density.

In this paper, we aim to achieve automatic melody harmonization with flexible harmonic rhythm. To generate chord progressions that rhythmically match the given melody, we first model the task by generating chords frame-by-frame instead of bar-by-bar. We then encode time signatures to establish the rhythmic relationships between melodies and chords. Based on [gamma sampling](https://arxiv.org/pdf/2205.06036.pdf), we further implement controllable harmonic density, which offers possibilities for customized harmonizations based on users' preferences. Our contributions are threefold. 1) We considered beat and key information, thus our model can handle any number of time signatures and key signatures in a piece, without being limited to specific notations (e.g., C major and 4/4). 2) AutoHarmonizer predicts chords frame-by-frame, which enables the generation of flexible harmonic rhythms. 3) With gamma sampling, users can adjust the harmonic density of model-generated chord progressions.

For more information, see our [arXiv paper](https://arxiv.org/abs/2112.11122) and [GitHub repo](https://github.com/sander-wood/autoharmonizer).  
