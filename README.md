# SCARGC.jl

A Julia implementation of **S**tream **C**lassification **A**lgorithm **G**uided by **C**lustering – **SCARGC** -, an algorithm to classify data streams in nonstationary environments with extreme verification latency. The considered scenario is the one where the actual labels of unlabeled data are never available as a guidance to update the classification model over time.

Documentation | Build
------------- | -----
[![][docs-stable-img]][docs-stable-url] | [![Build Status](https://travis-ci.org/MarinhoGabriel/SCARGC.jl.svg?branch=master)](https://travis-ci.org/MarinhoGabriel/SCARGC.jl) [![Coverage Status](https://coveralls.io/repos/github/MarinhoGabriel/SCARGC.jl/badge.svg?branch=master)](https://coveralls.io/github/MarinhoGabriel/SCARGC.jl?branch=master)


## Installation

The package can be installed using the Julia package REPL, by typing `]` in Julia terminal.
Once the package manager is opened, type:

```julia
pkg> add SCARGC
```

Or, if you prefer, using the `Pkg` API:

```julia
julia> using Pkg

julia> Pkg.add("SCARGC")
```

## References

Souza, V. M. A.; Silva, D. F.; Gama, J.; Batista, G. E. A. P. A.: **Data Stream Classification Guided by Clustering on Nonstationary Environments and Extreme Verification Latency**. SIAM International Conference on Data Mining (SDM), pp. 873-881, 2015

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://
