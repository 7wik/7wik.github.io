---
layout: post
title: Class conditional and generic disentangled representations
---

During my internship at IIT-H, I came across disentangled representations while going through the ICML paper [Disentangling by factorising](https://arxiv.org/pdf/1802.05983.pdf). In this work by Kim et al, it was proved that by minimizing a Total Correlation term through which their model encourages a factorial representation of the latent.

# Our problem

The above work learns a generalised set of latent factors for all the classes in a dataset. If we were interested in learning class-specific factors using Factor-VAE, we have to model as many models as there are classes which is quite hectic process. In order to address this problem, we proposed a new model. 

