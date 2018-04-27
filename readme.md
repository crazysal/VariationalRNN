## **What** the paper is about ?

The inclusion of latent random variables into the hidden state of a recurrent neural network (RNN) by combining the elements of the variational autoencoder. Use of high-level latent random variables of the variational RNN (VRNN) to model the kind of variability observed in highly structured sequential data such as natural speech.

**Target:** Learning generative models of sequences.

**Argument: **Complex dependencies cannot be modelled efficiently by the output probability models used in standard RNNs, which include either a simple unimodal distribution or a mixture of unimodal distributions.

**Assumptions: **Only concerned with  highly structured data having - 

* relatively high signal-to-noise ratio, variability observed is due to signal not noise.

* underlying factors of variation and the observed data are strongly correlated. 

** **

## **Why** is the architecture so powerful and how it is differentiated by other methods?

**Hypothesis :** Model variability should induce temporal dependencies across timesteps.

**Implement : **Like DBN models such as HMMs and Kalman filters, model the dependencies between the latent random variables across timesteps. 

**Innovation :** Not the first to propose integrating random variables into the RNN hidden state, the first to integrate the dependencies between the latent random variables at neighboring timesteps. Extend the VAE into a recurrent framework for modelling high-dimensional sequences.

## **What implementation **is available? (Setting up the code environment)

Most of the script files are written as pure Theano code, modules are implemented from a more [general framework written by original author](https://github.com/jych/cle), Junyoung Chung.  

Yoshua Bengio announced on Sept. 28, 2017, that[ development on Theano would cease](https://groups.google.com/d/msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ). **Theano is effectively dead.**

Many academic researchers in the field of deep learning rely on[ Theano](http://deeplearning.net/software/theano/), the grand-daddy of deep-learning frameworks, which is written in[ Python](https://darkf.github.io/posts/problems-i-have-with-python.html). Theano is a library that handles multidimensional arrays, like Numpy. Used with other libs, it is well suited to data exploration and intended for research.

Numerous open-source deep-libraries have been built on top of Theano, including[ Keras](https://github.com/fchollet/keras),[ Lasagne](https://lasagne.readthedocs.org/en/latest/) and[ Blocks](https://github.com/mila-udem/blocks). These libs attempt to layer an easier to use API on top of Theano’s occasionally non-intuitive interface. (As of March 2016, another Theano-related library,[ Pylearn2, appears to be dead](https://github.com/lisa-lab/pylearn2).)

Pros and Cons

* (+) Python + Numpy

* (+) Computational graph is nice abstraction

* (+) RNNs fit nicely in computational graph

* (-) Raw Theano is somewhat low-level

* (+) High level wrappers (Keras, Lasagne) ease the pain

* (-) Error messages can be unhelpful

* (-) Large models can have long compile times

* (-) Much "fatter" than Torch

* (-) Patchy support for pretrained models

* (-) Buggy on AWS

* (-) Single GPU

## What can you **improve**? (Your contribution to the existing code.)

Possible ideas : 

* Implementation in an entirely new framework like pytorch/tensorflow which is much more gpu friendly thus allowing for testing on larger more higher dimensional datasets. Like probable chess moves. 

* The VRNN contains a VAE at every timestep. However, these VAEs are conditioned on the state variable ht − 1 of an RNN. This addition will help the VAE to take into account the temporal structure of the sequential data.  Experimenting with a replay buffer and batch normalizations could probably have a huge impact.  

Advantages of experience replay:

* More efficient use of previous experience, by learning with it multiple times. This is key when gaining real-world experience is costly, you can get full use of it. The memory-learning updates are incremental and do not converge quickly, so multiple passes with the same data is beneficial, especially when there is low variance in immediate outcomes (reward, next state) given the same state, action pair.

* Better convergence behaviour when training a function approximator. Partly this is because the data is more like[ i.i.d.](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) data assumed in most supervised learning convergence proofs.

Disadvantage of experience replay:

* It is harder to use multi-step learning algorithms, such as Q(λ), which can be tuned to give better learning curves by balancing between bias (due to bootstrapping) and variance (due to delays and randomness in long-term outcomes)

## Describe the **Dataset** in use. Can you apply these methods to some other dataset?

**Comparison :** The proposed VRNN model against other RNN-based models – including a VRNNmodel without introducing temporal dependencies between the latent random variables.

**Tasks :**

* Modelling natural speech directly from the raw audio waveforms

* Modelling handwriting generation.

**Speech Modelling** : Directly model raw audio signals, represented as a sequence of 200-dimensional frames. Each frame corresponds to the real-valued amplitudes of 200 consecutive raw acoustic samples. Note that this is unlike the conventional approach for modelling speech, often used in speech synthesis where models are expressed over representations such as spectral features.

**Evaluation -**

* **Blizzard**: This text-to-speech dataset made available by the Blizzard Challenge 2013 contains 300 hours of English, spoken by a single female speaker.

* **TIMIT**: This widely used dataset for benchmarking speech recognition systems contains 6300 English sentences, read by 630 speakers.

* **Onomatopoeia2** : This is a set of 6, 738 non-linguistic human-made sounds such as coughing, screaming, laughing and shouting, recorded from 51 voice actors.

* **Accent**: This dataset contains English paragraphs read by 2, 046 different native and nonnative English speakers.

**Handwriting Modelling : **Each model learn a sequence of (x, y) coordinates together with binary indicators of pen-up/pen-down.

**Evaluation - **

* **IAM-OnDB:** dataset which consists of 13040 handwritten lines written by 500 writers.

Changes on this project could potentially work on any generic sequential datasets. 

Example : 

* [Sequential penstroke data of MNIST](https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/wiki/MNIST-digits-stroke-sequence-data) : The MNIST handwritten digit images transformed into a data set for sequence learning. This data set contains pen stroke sequences based on the original MNIST images.

* The most popular benchmarks for sequential data are [PTB ](https://catalog.ldc.upenn.edu/ldc99t42)and TIMIT, authors used TIMIT but not PTB.

