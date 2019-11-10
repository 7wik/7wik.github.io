---
layout: post
title: Class conditional and class generic disentangled representations
---

During my internship at IIT-H, I came across disentangled representations while going through the ICML paper [Disentangling by factorising](https://arxiv.org/pdf/1802.05983.pdf). In this work by Kim et al, Factor-VAE, modeling of the disentangled representations is done by encouraging the marginal distribution of representation to be factorial. They achieved this by minimizing the total correlation(TC) of the latent units- where we define the TC of a set of random variables $$z_1,z_2,...z_n$$ as 
\begin{equation}
  TC(z_1,z_2,...,z_n)= KL(p(z_1,z_2,...,z_n)||\prod_{i=1}^{n}p(z_i))
\end{equation} (Here $$KL$$ stands for Kullback-leibler divergence). They estimated TC by using [density ratio estimation](http://yosinski.com/mlss12/media/slides/MLSS-2012-Sugiyama-Density-Ratio-Estimation-in-Machine-Learning.pdf).  
# Problem

Factor-VAE learns a generalised set of factors of variation for all the classes in a dataset,i.e, using Factor-VAE, we can model only those factors which are common to all the classes. But, what if we want to learn class specific factors? or if we want to understand the class-specific properties? using Factor-VAE, we are forced to learn as many models as there are classes which is quite cumbersome, thus, it's not desirable. In order to address this problem, we proposed a solution.





