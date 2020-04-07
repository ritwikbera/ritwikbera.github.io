---
permalink: /research/
title: "Research"
author_profile: true
redirect_from: 
    - /work

---


## [PODNet: A Neural Network for Discovery of Plannable Options](https://arxiv.org/abs/1911.00171)

Learning from demonstration has been widely studied in machine learning but becomes challenging when the demonstrated trajectories are unstructured and follow different objectives. _PODNet or Plannable Option Discovery Network_, addresses how to segment an unstructured set of demonstrated trajectories for option discovery. This enables learning from demonstration to perform multiple tasks and plan high-level trajectories based on the discovered option labels. PODNet combines a custom categorical variational autoencoder, a recurrent option inference network, option-conditioned policy network, and option dynamics model in an end-to-end learning architecture.

### Behavior discovery for point robot in custom built CircleWorld environment.

<img src="/assets/img/2.png" width="300" height="300">
<img src="/assets/img/4.png" width="300" height="300">
<img src="/assets/img/5.png" width="300" height="300">
<img src="/assets/img/6.png" width="300" height="300">


*The network had no prior idea about clockwise and counterclockwise motions and discovered two dynamically unqiue sub-behaviors by itself (inferred in red and blue) which happen to correspond well to human recognition of clockwise and counterclockwise motion as two sub-behaviors for this point robot.*