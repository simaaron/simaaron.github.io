---
published: true
layout: post
description: Data analysis competition solution
modified: {}
tags: 
  - recurrent neural networks
  - kaggle
  - rainfall
  - polarmetric radar
  - deep learning
image: 
  feature: "abstract-3.jpg"
  credit: dargadgetz
  creditlink: "http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/"
---


I recently participated in the Kaggle-hosted data science competition [_How Much Did It Rain II_](https://www.kaggle.com/c/how-much-did-it-rain-ii) where the goal was to predict a set of hourly rainfall levels from series of weather radar measurements. I came in _last_! I describe my approach in this blog post.

My research lab supervisor Dr John Pinney and I were in the midst of developing a set of deep learning tools for our current research programme when this competition came along. Due to some overlaps in the statistical tools and data sets (neural networks and variable-length sequences, in particular) I saw it as a good opportunity to validate some of our ideas in a different context (at least that is my _post hoc_ justification for the time spent on this competition!). In the near future I will post about the research project and how it relates to this problem. 

One of things which inspired me to write this up, other than doing well in the competition, was the awesome [blog post](http://benanne.github.io/2015/03/17/plankton.html) by Sander Dieleman where he describes his team's winning approach in another Kaggle competition I took part in. I hope this post will be as useful to others as his was to me.

## Introduction

### The problem

The goal of the competition was to predict a set of hourly rain gauge measurements--recorded over several months in 2014 over the US midwestern corn-growing states--from sequences of polarmetric weather radar data. These radar readings represent instantaneous snapshots of precipitation types and levels and is used to estimate rainfall levels over a wider area (e.g. nationwide) than can practically be covered by rain gauges.

For the contest, each radar measurement is condensed into 22 features. These include the minutes past the top of the hour that the radar observation was carried out, the distance of the rain gauge from the radar, and various reflectivity and differential phase readings of both the vertical column above and the areas surrounding the rain gauge. Up to 19 radar records are given per hourly rain gauge reading (and as few as one); intriguingly the number of radar measurements provided should itself contain some information on the rainfall levels as it is apparently not uncommon for meteorologists to request multiple radar sweeps when there are fast-moving storms.

The preditions were evaluated based on the mean absolute error (MAE) to the actual rain gauge readings. 


### The solution: Recurrent Neural Networks
The prediction of cumulative values from variable-length sequences of vectors with a 'time' component is virtually identical to the so-called _Adding Problem_--a toy sequence regression task that is designed to demonstrate the power of recurrent neural networks in learning long-term dependencies (see [Le et. al.](http://arxiv.org/abs/1504.00941) Sec. 4.1, for a recent example): 

<figure>
<center>
<img src="/images/RNN_adding.png" alt="The Additional Problem" width="450">
</center>
<figcaption>
The prediction target of 1.7 is obtained by summing up the numbers in the top row where the corresponding number in the bottom row is equal to one (i.e. the green boxes). The regression task is to infer this generative model from a training set consisting of random sequences of arbitrary lengths and their targets.
</figcaption>
</figure>

In our rainfall prediction problem, the situation is somewhat less trivial as there is still the additional step of inferring the rainfall 'values' (the top row) from radar measurements. Furthermore, instead of binary 1/0 values (bottom row) one has continuous time readings between 0 and 60 minutes that have somewhat different roles. Nevertheless, the underlying structural similarities are compelling enough to suggest that RNN would be well-suited for the problem. 

In the [previous version](https://www.kaggle.com/c/how-much-did-it-rain) of this contest (which I did not participate in), gradient boosting was the undisputed star of the show; neural networks, at least to my knowledge, were not deployed with much success. If RNNs could work as well as I hoped it would, then I might have a chance of coming up with a unique, if unreasonably effective, solution (i.e. win!).

For a overview of RNNs, the [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy is as good a general introduction to the subject as you will find anywhere.


### Software and hardware
I used [Python](https://www.python.org/) with [Theano](http://deeplearning.net/software/theano/) throughout, relying heavily on the [Lasagne](http://lasagne.readthedocs.org/en/latest/index.html) layer classes to build the RNN architectures. Additionally, I used [scikit-learn](http://scikit-learn.org/stable/) to implement the cross-validation testing splits, [pandas](http://pandas.pydata.org/) and [NumPy](http://www.numpy.org/) to process and format the data and submission files.

## Data pre-processing and augmentation

### Data challenges


### Data augmentation 


<figure>
<center>
<img src="/images/RNN_01.png" alt="Dropin augmentation" width="450">
</center>
<figcaption>
"Dropin" augmentations of a length-5 sequence to length-8 sequences. The number labels are the timestamps of the given data points (minutes past the hour) 
</figcaption>
</figure>

## RNN architecture


### Design evolution


### Nonlinearities


## Training

### Local validation

### Initialisation

### Regularisation


## Model ensembles

### Test-time augmentation

### Averaging models


##
