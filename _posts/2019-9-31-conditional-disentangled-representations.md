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

# Assumptions and approach

We assume that we have a dataset $$\mathbb{D}$$ of samples $$(x,y)$$ with $$n$$ classes where $$x$$ is our observation and $$y$$ is it's one-hot encoded class label belonging to$$\{y \in \{0,1\}^{n} \mid \sum_i(y_i)=1\}$$. We consider a VAE kind of setup for our experiments and problem solving. Since we are interested in learning disentangled representations which take the class information into consideration, the obvious choice is to explicitly condition the encoder with the class label for learning representations with cues from the class label.

# naive-model
We consider the set of latent variables to be denoted by $$\textbf{z}$$ and we define a posterior $$ q_{\phi}(z|x,y) $$, a prior $$ p(z|y) $$ and likelihood $$ p_{\theta}(x|z,y) $$ This model's framework is similar to a fully supervised conditional VAE setup, where we try to optimize the variational lower bound 
\begin{equation}
  p(x|y) \geq E_{q_{\theta}(z|x,y)}[p(x|z,y)] - KL(q_{\theta}(z|x,y)||p(z))      
\end{equation}
But in our case since we want to learn disentangled representations, like in [$$\beta-$$VAE](https://openreview.net/references/pdf?id=Sy2fzU9gl), we have an additional weight on the KL-divergence term to ensure disentanglement of the variables. 
The problem with this objective is quite clear, similar to that of $$\beta-$$VAE's the additional weight that we put on KL-divergence term is adding on to minimizing the mutual information between $$z$$ and the joint variable $$(x,y)$$ w.r.t the variational joint distribution as well. Hence, even though the $$z$$ gets disentangled, the latent code's knowledge about the data is minimized in every step. This can be easily shown following the [Makhzani et al](https://arxiv.org/pdf/1706.00531.pdf) and proof in [Factor-VAE](https://arxiv.org/pdf/1802.05983.pdf).
\begin{align}
  E_{(x,y)\sim D(x,y)}[KL(q_{\theta}(z|x,y)||p(z))] = I(z;(x,y))+KL(q(z)||p(z))
\end{align}
We generally take standard gaussian as $$p(z)$$, so the representation $$q(z)$$ gets disentangled because of the weight term. To restrict the problem aroused by $$ I(z;(x,y)) $$, we can put a cap on the $$KL$$ term in this objective with a gradually increasing positive term $$ C_z $$ and modify our objective similar to that of [burgess et al's](https://arxiv.org/pdf/1804.03599.pdf). Using an approach like this simply narrows down our intent for learning meaningful representations. We expect to learn factors both which are and which are not affected by class-information. To solve this problem, we can use the following model, the details of which are explained in the next subsection.

# Proposed Model
![Gaussian]({{ site.baseurl }}/images/model-2.jpg "Proposed model")
We propose a model which is a modification to the previous model's framework which allows us to model both class-dependent and class-independent factors. Let $$\textbf{z}$$,$$\textbf{w}$$ denote the set of class-dependent and class-independent variables respectively. We define a joint posterior $$ q_{\phi}(z,w|x,y) $$, a prior $$ p(z,w|y) $$ and likelihood $$ p_{\theta}(x|z,w,y) $$. With this premise, the conventional $$ \beta $$-VAE kind of objective is,
\begin{equation}
    L(\theta,\phi)= E_{q_{\phi}(z,w|x,y)}[p_{\theta}(x|z,w,y)]-\beta .KL(q_{\phi}(z,w|x,y)||p(z,w|y))
\end{equation}
Since we have assumed that $$ z \perp y $$ and $$ z \perp w $$, we rewrite this objective as
\begin{equation}
    L(\theta,\phi)= E_{q_{\phi}(z,w|x,y)}[p_{\theta}(x|z,w,y)]-\beta .KL(q_{\phi}(z|x)||p(z|y)) - \gamma .KL(q_{\phi}(w|x,y)||p(w|y)) 
\end{equation}
Taking inspiration from [burgess et al](https://arxiv.org/pdf/1804.03599.pdf), we put a cap on the KL-divergence terms so that there is not much information about the data lost. With this our modified objective is
\begin{align}
  L_{VAE}= E_{q_{\phi}(z,w|x,y)}[p_{\theta}(x|z,w,y)]-\beta|KL(q_{\phi}(z|x)||p(z|y)) - C_z| - \gamma|KL(q_{\phi}(w|x,y)||p(w|y)) - C_w| 
\end{align}

# Minimizing class dependence
We have assumed that $$z \perp y$$ and we wanted most information related to $$y$$ to dwell in $$w$$, so we explicitly minimize the mutual information between $$z$$ and $$y$$. We do this by making use of a classifier network. If $$z \perp y$$, the classifier misclassifies a given $$z$$. For this purpose, we define an augmented loss to be maximized by the network using a binary-cross-entropy.
\begin{align}
&   L_{class}(z)= y.log(\sigma(h_{\psi}(z))) + (1-y).log(1-\sigma(h_{\psi}(z)))
\end{align}
Since the classifier must also be capable of predicting the correct class label given $$z$$, the classifier's parameters are also to be trained to achieve the below defined objective-
\begin{alignat}{2}
 \min_{\psi}        \quad &&  L_{class}
\end{alignat}
So, our overall objective is to solve the below multi-step optimization problem where we train the parameters of encoder and decoder by solving the maximization problem in the first-step and train the parameters of the classifier in the next step while solving the minimization problem-
\begin{align}
  &  \max_{\theta,\phi} \quad && L(\theta,\phi)+ L_{class} 
\end{align} 
\begin{align}
  &  \min_{\psi}  \quad && L_{class}
\end{align} 
