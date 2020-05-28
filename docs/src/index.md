# SCARGC.jl

*A Julia Implementation of Stream Classification Algorithm Guidded by Clustering*

[![Build Status](https://travis-ci.org/MarinhoGabriel/SCARGC.jl.svg?branch=master)](https://travis-ci.org/MarinhoGabriel/SCARGC.jl)
[![Coverage Status](https://codecov.io/gh/MarinhoGabriel/SCARGC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MarinhoGabriel/SCARGC.jl)
[![License File](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/MarinhoGabriel/SCARGC.jl/blob/master/LICENSE)

[SCARGC.jl](https://github.com/MarinhoGabriel/SCARGC.jl) is a Julia implementation 
of Stream Classification Algorithm Guided by Clustering - SCARGC -, a data stream 
classifier in non-stationary environments with extreme verification latency. The 
implementation was made using the paper as base and the official implementation 
of SCARGC, in MatLab.

## Overview

Most of research data on data stream classification with concept drift, assumes 
that there's a availability of the labels, instantly or with some delay after 
the classification occurs. Then, these methods verify if the model is outdated
based on the classification information obtained on the recent data.

[SCARGC.jl](https://github.com/MarinhoGabriel/SCARGC.jl) is an algorithm that
make predictions with a EVL scenary, where no labeled data is received after the 
initialization of the model. So, for example, we receive a small amount of labeled
data, build a classifier model with this data and, after that, there's no labeled
data "arriving". We just predict and update the model to get a better result in
the next iteration.

## Installation

In the Julia's package manager, type:

```julia-REPL
pkg> add SCARGC
```

## Tutorials

You can find some notebooks explaining how to use the package in 
[experiments](https://github.com/MarinhoGabriel/SCARGC.jl/tree/master/experiments).

## Quick example

We can see below a example of using the SCARGC.jl package to predict labels in
datasets.

!!! note
    It's important to say that the dataset used in the example is a synthetic 
    dataset, so we already have the instances's label. Using it, we can say the
    accuracy of the algorithm. If you're using it with some real application, 
    with no data, the accuracy calculus isn't going to happen.

```@example overview
using SCARGC

# loading the dataset
dataset = "../../src/datasets/synthetic/1CHT.txt"
data = SCARGC.extractValuesFromFile(dataset, 16000, 3)

# predicting data
labels, accuracy = SCARGC.scargc_1NN(data, 0.3125, 300, 2)

# then, we can print the accuracy
println("Accuracy: ", accuracy, "%")
```

## Results

Today, the results of [SCARGC.jl](https://github.com/MarinhoGabriel/SCARGC.jl) are
equivalent to the first implemented, in MatLab.

In the picture below, the blue bars represent the original SCARGC and the
black ones represent SCARGC.jl results on those datasets (represented in X asis).


![Result](https://github.com/MarinhoGabriel/SCARGC.jl/blob/master/results/result.jpeg?raw=true)

## References

[Souza, V. M. A.; Silva, D. F.; Gama, J.; Batista, G. E. A. P. A.: **Data Stream Classification Guided by Clustering on Nonstationary Environments and Extreme Verification Latency**. SIAM International Conference on Data Mining (SDM), pp. 873-881, 2015](https://repositorio.inesctec.pt/bitstream/123456789/5325/1/P-00K-AN4.pdf)