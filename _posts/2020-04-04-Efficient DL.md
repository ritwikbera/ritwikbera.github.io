---
title: 'Efficient Deep Learning'
date: 2020-04-04
permalink: /posts/2020/04/efficientdl/
tags:
  - ML
---

This is a sample blog post. Lorem ipsum I can't remember the rest of lorem ipsum and don't have an internet connection right now. Testing testing testing this blog post. Blog posts are cool.

Making Deep Learning Efficient
======


__Why do we care about hardware when it comes to deep learning ?__
There are a number of practical reasons to warrant more efficient DL systems:
* _Lower power consumption_: This has an impact on range for applications like self-driving cars.
* _Lower latency_: Anything is fast is welcome. Models whose weights can be saved on fast on-chip memory/registers have fast inference as on-chip memory has the lowest access time.
* _Saved real estate from no heat sink requirement_: The ability to run on more compact hardware enables deployment of ML systems on UAVs, smartphones etc. Usually models requiring frequent memory accesses run hotter.
* _Smaller download size for OTA update_: Software updates for ML applications can be less bulky and have more penetration.
* _Ability to run on cheaper hardware_: A no brainer, this is always good to have for wide adoption.

The Hardware Angle
------

__So why does specialised hardware work so well ?__
A number of reasons,
* ML specific hardware can have improvements such as higher on-chip memory (like register memory on GPUs) to be able to store weights etc. This reduces the need for expensive accesses to off-chip memory (global memory) and improves throughput and reduces latency.

In deep learning applications, memory bandwidth is constrained due to constant reloading of the next layer, storing activations in global memory and performing the next set computations. Better hardware (than general purpose GPUs) can alleviate this. If activations can also be stored in on-chip memory (alongwith weights), its even better !.

* Purpose-built custom hardware provides extensive performance improvements and speedups owing to the the fact that most of deep learning inference and training is [just matrix multiplications](https://danieltakeshi.github.io/2017/01/21/understanding-higher-order-local-gradient-computation-for-backpropagation-in-deep-neural-networks/). It holds especially [true for CNNs](https://danieltakeshi.github.io/2019/03/09/conv-matmul/).

Hardware like TPUs are ASICs built for matrix multiplication only and bypass the von Neumann bottleneck. For example, they might perform 64x64 matrix multiplications in a single pass with memory access required only once.

![TPU](https://2.bp.blogspot.com/-yhjY3pc6oow/WLRn2z4mPBI/AAAAAAAACcU/t_EAR6QMwQQkTBPftJQEonaB2DMbRXmXwCLcB/s1600/Screen%2BShot%2B2017-02-27%2Bat%2B9.54.12%2BAM.png)
*An example of a case where some primitive tensor ops (kernels) can be fused to avoid the need to store intermediate results. Source: https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html*

* Hardware like TPUs also support specialised compilation like XLA. XLA fuses multiple CUDA kernel ops depending upon the program and runs them at once, instead of each kernel being called separately. (See [here](https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html)). Such fused implementations minimize global memory accesses thus saving precious memory bandwidth. This also reduces overhead associated with multiple kernel launches.