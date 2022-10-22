---
layout: default
title: Controllable Harmonic Density
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

## Controllable Harmonic Density

Based on the assumption that some attributes of the generated text are closely related to the number of occurrences of some tokens, Wu et al. proposed [gamma sampling](https://arxiv.org/pdf/2205.06036.pdf) for controlling language models. It achieved controllable text generation by scaling the probability $p$ of the attribute-related token during generation time:

$$p^{\mathcal{A}}_{out}=p_{in}^{\mathcal{A}tan(\frac{\pi \Gamma}{2})}, \\ p^{a}_{out}=p^{a}_{in}\cdot \frac{p^{\mathcal{A}}_{out}}{p^{\mathcal{A}}_{in}},\quad \forall a\in \mathcal{A}, \\ p^{n}_{out}=p^{n}_{in} \cdot (1 + \frac{p^{\mathcal{A}}_{in}-p^{\mathcal{A}}_{out}}{p^{\backslash \mathcal{A}}_{in}}),\quad \forall n\notin \mathcal{A},$$

where $\Gamma\in$[0,1] is the user-controllable parameter, $p_{in/out}$ is the input/output probability, $\mathcal{A}$ is the set of attribute-related tokens and $p^{\mathcal{A}}$ is the sum of their probabilities, while $p^{\backslash \mathcal{A}}$ is the sum of the probabilities of tokens that are not in $\mathcal{A}$. When $\Gamma=0.5$, there is no change in the probability distribution, while when $\Gamma<0.5$, the probabilities of the attribute-related tokens increase and vice versa.

To achieve controllable harmonic density, when generating the chord $c_{t}$ at time step $t$, we select the previously generated chord token $c_{t-1}$ as the attribute-related token. When $\Gamma>0.5$, AutoHarmonizer tends to generate chords different from $c_{t-1}$, thus the switching of chords becomes more frequent, making it tend to generate denser chord progressions, and vice versa.  

In most cases, the output probability is high for essential chords (i.e., tonic, dominant, and subdominant chords) and low for non-essential chords. Depressing the frequency of chord switching makes it likely to omit non-essential chords, and vice versa, generating more non-essential chords.
