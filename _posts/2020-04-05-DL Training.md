---
title: 'DL Training'
date: 2020-04-05
permalink: /posts/2020/04/dltraining/
tags:
  - ML
---

Thoughts on deep learning training.
======

## Tips for improving training performance

* __Mixed Precision Training__

* __Gradient Checkpointing__

* __Second-order Optimization__

* __BatchNorm__


## Distributed Training

* __Data vs Model Parallelism__
* __Asynchronous Optimization__
* __Reduce__


## Other Techniques

* __Population-Based Training__ for hyperparameter search

Used by Waymo in training their NN models and searching for good data-augmentation policies to train their vision pipeline.

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