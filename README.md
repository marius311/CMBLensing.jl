# CMBLensing.jl

This repository contains tools written in [Julia](https://julialang.org/) to analyze the gravitationally lensed Cosmic Microwave Background. 

Some things this code can do:

* Lense flat-sky temperature and polarization maps using the following algorithms:
    * Taylor series expansion to any order
    * The `Taylens` algorithm ([Næss & Louis 2013](https://arxiv.org/abs/1307.0719))
    * The `LenseFlow` algorithm ([Millea, Anderes, & Wandelt 2017](https://arxiv.org/abs/1708.06753))
* Compute the quadratic estimate of $\phi$ given some data ([Hu & Okamoto 2003](https://arxiv.org/abs/astro-ph/0111606))
* Compute best-fit of $\mathcal{P}(f,\phi\,|\,d)$, i.e. the joint maximum a posteriori estimate of the lensing potential and CMB fields, and draw Monte-Carlo samples from this poterior, with the option to sample over cosmological parameters as well ([Millea, Anderes, & Wandelt 2017](https://arxiv.org/abs/1708.06753))
* Compute best-fit of $\mathcal{P}(\phi\,|\,d)$, i.e. the marginal maximum a posteriori estimate of the lensing potential ([Carron & Lewis 2017](https://arxiv.org/abs/1704.08230))

## Documentation

The best place to get started is to read the [documentation](https://cosmicmar.com/CMBLensing.jl) (which is very much a work-in-progress, many things this package can do are not documented yet, but are planned to be added soon). 

Most of the pages in the documentation are Jupyter notebooks which you can find in [this folder](https://github.com/marius311/CMBLensing.jl/tree/master/docs/src), which you can run yourself. 

## Requirements

* Julia 1.0 or higher
* Python 3 + matplotlib (used for plotting)
* (optional) [pycamb](https://github.com/cmbant/CAMB) to be able to generate $C_\ell$'s
* (optional) [healpy](https://github.com/healpy/healpy) for experimental curved sky support

## Native installation

To install the Julia package locally, run:

```julia
pkg> dev https://github.com/marius311/CMBLensing.jl
```

(type `]` at the Julia REPL to reach the `pkg>` prompt)

## Run via Docker

Also provided is a Docker container which includes a Jupyter notebook server and all the dependencies to run and use `CMBLensing.jl`. This is probably the quickest way to get up and running including all of the optional dependencies (you just need Docker and docker-compose on your system). To launch the Jupyter notebook, clone this repository and run the following from the root directory,

```sh
docker-compose pull
docker-compose up
```

The first time you run this, it will automatically download the (~500Mb) container from the Docker hub. The command will prompt you with the URL which you should open in a browser to access the notebook.

To run the notebook on a different port than the default `8888`, do `PORT=1234 docker-compose up` where `1234` is whatever port number you want.

You can also build the container locally by replacing `docker-compose pull` with `docker-compose build` above.
