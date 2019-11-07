---
layout: post
title: Class conditional and generic disentangled representations
---

During my internship at IIT-H, I came across disentangled representations while going through the ICML paper [Disentangling by factorising](https://arxiv.org/pdf/1802.05983.pdf). In this work by Kim et al, Factor-VAE, modeling of the disentangled representations is done by encouraging the marginal distribution of representation to be factorial. They achieved this by minimizing the total correlation(TC) of the latent units- where we define the TC of a set of random variables $$z_1,z_2,...z_n$$ as

$$TC(z_1,z_2,...,z_n)= KL(p(z_1,z_2,...,z_n)||\pi_{i=1}^{n}p(z_i))$$

# Our problem

The above work learns a generalised set of latent factors for all the classes in a dataset. If we were interested in learning class-specific factors using Factor-VAE, we have to model as many models as there are classes which is quite hectic process. In order to address this problem, we proposed a new model. 

