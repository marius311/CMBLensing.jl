# CMBLensing.jl


[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cosmicmar.com/CMBLensing.jl/) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marius311/CMBLensing.jl/master?urlpath=lab) [![Build Status](https://travis-ci.org/marius311/CMBLensing.jl.svg?branch=master)](https://travis-ci.org/marius311/CMBLensing.jl)

This repository contains tools written in [Julia](https://julialang.org/) (and easily callable from Python) to analyze the gravitationally lensed Cosmic Microwave Background. 

Some things this code can do:

* Lense flat-sky temperature and polarization maps using the following algorithms:
    * Taylor series expansion to any order
    * The `Taylens` algorithm ([Næss & Louis 2013](https://arxiv.org/abs/1307.0719))
    * The `LenseFlow` algorithm ([Millea, Anderes, & Wandelt 2017](https://arxiv.org/abs/1708.06753))
* Compute the quadratic estimate of $\phi$ given some data ([Hu & Okamoto 2003](https://arxiv.org/abs/astro-ph/0111606))
* Compute best-fit of $\mathcal{P}(f,\phi\,|\,d)$, i.e. the joint maximum a posteriori estimate of the lensing potential and CMB fields, and draw Monte-Carlo samples from this poterior, with the option to sample over cosmological parameters as well ([Millea, Anderes, & Wandelt 2017](https://arxiv.org/abs/1708.06753))
* Compute best-fit of $\mathcal{P}(\phi\,|\,d)$, i.e. the marginal maximum a posteriori estimate of the lensing potential ([Carron & Lewis 2017](https://arxiv.org/abs/1704.08230))

## Documentation

The best place to get started is to read the [documentation](https://cosmicmar.com/CMBLensing.jl/) (which is very much a work-in-progress, many things this package can do are not documented yet, but are planned to be added soon). 

Most of the pages in the documentation are Jupyter notebooks, and you can click the "launch binder" link at the top of each page to launch a Jupyterlab server running the notebook in your browser (courtesy of [binder](https://mybinder.org/)). You can also find the notebooks in [this folder](https://github.com/marius311/CMBLensing.jl/tree/gh-pages/src) if you want to run them locally (which will usually lead to higher performance).

## Installation

### Requirements

* Julia 1.0 or higher
* Python 3 + matplotlib (used for plotting)
* (recommended) [pycamb](https://github.com/cmbant/CAMB) to generate $C_\ell$'s
* (optional) [healpy](https://github.com/healpy/healpy) for experimental curved sky support

### Native installation

To install the Julia package locally, run:

```juliapkg
pkg> add https://github.com/marius311/CMBLensing.jl#master
```

(type `]` at the Julia REPL to reach the `pkg>` prompt)

### Docker installation

Also provided is a Docker container which includes a Jupyterlab server and all the recommended and optional dependencies to run and use `CMBLensing.jl`. Launch this container with:

```sh
git clone https://github.com/marius311/CMBLensing.jl.git
cd CMBLensing.jl
docker-compose pull
docker-compose up
```

The first time you run this, it will automatically download the (~500Mb) container from the Docker hub. The command will prompt you with the URL which you should open in a browser to access the notebook.

To run the notebook on a different port than the default `8888`, do `PORT=1234 docker-compose up` where `1234` is whatever port number you want.

You can also build the container locally by replacing `docker-compose pull` with `docker-compose build` above.
