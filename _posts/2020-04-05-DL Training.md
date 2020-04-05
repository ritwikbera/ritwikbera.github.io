---
title: 'DL Training'
date: 2020-04-05
permalink: /posts/2020/04/dltraining/
tags:
  - ML
---

Thoughts on deep learning training.
======

## Improving Single Device Training Performance

* __Mixed Precision Training__

    Mixed precision training speeds up training by using 16 bit weights while maintaining a 32 bit copy of weights for high precision inference.
    During training, weights are rounded off to 16 bit representations and gradients scaled up to make them significant in 16 bit precision. During the update step on the 32 bit master copy of the weights, the gradients are scaled down to restore their original value.
    Using 16 bit weights for training leads to a lower memory requirement for storing activations.

* __Gradient Checkpointing__

    Instead of storing all intermediate activations (to be used during the backward pass), one mays store only some of the intermediate activations (checkpoints) and recompute the others (by doing a forward pass from the checkpointed activations) everytime they are needed.

    Gradient checkpointing can be a trade-off that reduces memory consumption but increases compute requirement.

* __Second-order Optimization__

    First-order optimization methods take time to converge, as first derivates often do not capture the loss surface very well. Second-order optimization methods are more accurate but they require costly matrix inversion operations. Techniques like _Kronecker Factored Approximate Curvature K-FAC_ enable inexpensive second-order optimization as the creatively factorize out large matrices into smaller block matrices. For more details visit [here](https://towardsdatascience.com/introducing-k-fac-and-its-application-for-large-scale-deep-learning-4e3f9b443414)

* __BatchNorm__:

    Batch norm reduces the effects of second-order interactions between weights in different layers, thus stabilising gradient values and speeding up convergence.

## Improving Multi Device Training Performance

* __Data vs Model Parallelism__

    _Data Parallelism_ is preferred when we have lots of data that can't fit on a single device and exchanging weight gradients is relatively easier (especially in case convolutions where kernels have relatively few parameters). 

    On the other hand, _Model Parallelism_ is preferred when there are too many parameters to be fit on a single device and exchange activation gradients is easier.

* __Asynchronous Optimization__

    Asynchronous gradient descent with multiple workers is the bedrock of techniques like _A3C_ in reinforcement learning. The parameters to be updated are stored in shared memory and updated by each worker asynchronously. Since each worker has access only to a subset of the whole training data, the gradients are not very accurate and are noisy. However, gradients in deep learning are anyways noisy and this technique provides a decent speedup.

* __Reduce__

    Anytime distributed training happens, there is a need to accumulate all gradients from all GPUs via a _reduce_ operation. For large models, this can be a bottleneck step if gradients are accumulated sequentially from each GPU or if they cause latency issues due to lower inter-GPU memory bandwidth. There are a number of creative _reduce_ techniques that avoid the issues. Two famous ones are mentioned below.
        
    _Baidu's RingReduce_ descriptively written about [by Andrew Gibiansky](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/).  You can find my PyTorch-based implementation of _Baidu's RingReduce_ [here](https://github.com/ritwikbera/RingReduce).

    _Reduction Tree_ as described in the [_FireCaffe_ paper](https://arxiv.org/abs/1511.00175) by _Iandola et. al._. It is essentially a binary tree performing reduce operations in parallel (each process handles two GPU nodes at a time ) with _O (log N)_ complexity.

## Other Techniques

Finding good hyperparameters is essential for pain-free training routines. A relatively new method developed by _DeepMind_ is _Population Based Training (PBT)_.

* __Population-Based Training__

    Population-based training is essentially employing evolutionary algorithms to hyperparameter search wherein an ensemble of models is trained with each having their own set of hyperparameters that keeps evolving, transferring after certain intervals through the training procedure. It does seem to be the most efficient AutoML/Neural Architecture Search method so far.
    More recently it has been used by Waymo in training their NN models and searching for good [data-augmentation policies](https://blog.waymo.com/2020/04/using-automated-data-augmentation-to.html) to train their vision pipeline.

    ![pbt](/assets/img/pbt.png)
    *An overview of how top performing models keep mutating and the bottom performs adopt the top performers' parameters.Credits: DeepMind*

    A crude pseudocode to implement it is provided below.

```python

def hparam_search(models, K, perturbation):
    N = len(models)
    while max(scores) < MAX_SCORE:
        start training N different models
        ranked_models = sorted(models, lambda model: score(model))
        top_models = ranked_models[:K]
        bottom_models = ranked_models[K:]
        
        for model in bottom_models:
            top_model = random.choice(top_models)
            model['state_dict'] = top_model['state_dict']
            model['optimizer_state'] = top_model['optimizer_state']

        for hparam in hparams:
            for model in top_models:
                model[hparam] += perturbation
```