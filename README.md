# CMBLensing.jl

[![Build Status](https://travis-ci.org/marius311/CMBLensing.jl.svg?branch=master)](https://travis-ci.org/marius311/CMBLensing.jl)


This repository contains tools written in [Julia](https://julialang.org/) to analyze the gravitationally lensed Cosmic Microwave Background. 

Some things this code can do:

* Lense temperature and polarization maps using the following algorithms:
    * Taylor series expansion to any order
    * The `Taylens` algorithm ([Næss & Louis 2013](https://arxiv.org/abs/1307.0719))
    * The `LenseFlow` algorithm ([Millea, Anderes, & Wandelt 2017](https://arxiv.org/abs/1708.06753))
* Compute the quadratic estimate of ϕ given some data ([Hu & Okamoto 2003](https://arxiv.org/abs/astro-ph/0111606))
* Compute best-fit of 𝓟(f,ϕ|d), i.e. the joint maximum a posteriori estimate of the lensing potential and CMB fields, and draw Monte-Carlo samples from this poterior, with the option to sample over cosmological parameters as well ([Millea, Anderes, & Wandelt 2017](https://arxiv.org/abs/1708.06753))
* Compute best-fit of 𝓟(ϕ|d), i.e. the marginal maximum a posteriori estimate of the lensing potential ([Carron & Lewis 2017](https://arxiv.org/abs/1704.08230))

To get started with this code, you can first look at [this](docs/joint_MAP_example.ipynb) Jupyter notebook demonstrating a toy joint MAP analysis. 


## Requirements

* Julia 1.0 or higher
* (optional) Python 3 & [pycamb](https://github.com/cmbant/CAMB) to be able to generate Cℓ's
* (optional) Python 3 & [healpy](https://github.com/healpy/healpy) for experimental curved sky support

Note: all non-Julia dependencies are optional, so as long as you've got Julia running, you can run CMBLensing.jl. 

## Native installation

To install the Julia package locally, run:

```julia
pkg> dev https://github.com/marius311/CMBLensing.jl
```

(type `]` at the Julia REPL to reach the Julia `pkg>` prompt)

## Run via Docker

Also provided is a Docker container which includes a Jupyter notebook server and all the dependencies to run and use `CMBLensing.jl`. This is probably the quickest way to get up and running including all of the optional dependencies (you just need Docker and docker-compose on your system). To launch the Jupyter notebook, clone this repository and run the following from the root directory,

```sh
docker-compose pull
docker-compose up
```

The first time you run this, it will automatically download the (~500Mb) container from the Docker hub. The command will prompt you with the URL which you should open in a browser to access the notebook.

To run the notebook on a different port than the default `8888`, do `PORT=1234 docker-compose up` where `1234` is whatever port number you want.

You can also build the container locally by replacing `docker-compose pull` with `docker-compose build` above.
