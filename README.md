# CMBLensing.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cosmicmar.com/CMBLensing.jl/stable) [![Gitter](https://badges.gitter.im/CMBLensing-jl/community.svg)](https://gitter.im/CMBLensing-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

[![](https://img.shields.io/badge/source-github-blue)](https://github.com/marius311/CMBLensing.jl)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marius311/CMBLensing.jl/gh-pages?urlpath=lab) 
[![](https://github.com/marius311/CMBLensing.jl/workflows/runtests/badge.svg)](https://github.com/marius311/CMBLensing.jl/actions?query=workflow%3Aruntests+branch%3Amaster) [![](https://github.com/marius311/CMBLensing.jl/workflows/docbuild/badge.svg)](https://github.com/marius311/CMBLensing.jl/actions?query=workflow%3Adocbuild+branch%3Amaster) 


CMBLensing.jl is a next-generation tool for analysis of the lensed Cosmic Microwave Background. It is written in [Julia](https://julialang.org/) and transparently callable from Python.


At its heart, CMBLensing.jl maximizes or samples the Bayesian posterior for the CMB lensing problem. It also contains tools to quickly manipulate and process CMB maps, set up modified posteriors, and take gradients using automatic differentation.

### Highlights
* Fully Nvidia GPU compatible (speedups over CPU are currently 3x-10x, depending on the problem size and hardware).
* Automatic differentation (via [Zygote.jl](https://fluxml.ai/Zygote.jl/)) provides for-free gradients of your custom posteriors.
* Includes the following algorithms to lense a map:
    * `LenseFlow` ([Millea, Anderes, & Wandelt 2017](https://arxiv.org/abs/1708.06753))
    * `Taylens` ([Næss & Louis 2013](https://arxiv.org/abs/1307.0719))
    * Taylor series expansion to any order
    * Bilinear interpolation
* Maximize and sample $\mathcal{P}(f,\phi,\theta\,|\,d)$, the joint maximum a posteriori estimate of the lensing potential, $\phi$, the  temperature and/or polarization fields, $f$, and cosmological parameters, $\theta$ ([Millea, Anderes, & Wandelt 2017](https://arxiv.org/abs/1708.06753), [Millea, Anderes, & Wandelt 2020](https://arxiv.org/abs/2002.00965))
* Maximize $\mathcal{P}(\phi\,|\,d,\theta)$, i.e. the marginal maximum a posteriori estimate of the lensing potential, $\phi$, at fixed cosmological parameters, $\theta$ ([Carron & Lewis 2017](https://arxiv.org/abs/1704.08230))
* Do basic quadratic estimation of $\phi$ ([Hu & Okamoto 2003](https://arxiv.org/abs/astro-ph/0111606))

## Documentation

The best place to get started is to read the [documentation](https://cosmicmar.com/CMBLensing.jl/) (which is a work-in-progress, but contains many useful examples). 

Most of the pages in the documentation are Jupyter notebooks, and you can click the "launch binder" link at the top of each page to launch a Jupyterlab server running the notebook in your browser (courtesy of [binder](https://mybinder.org/)). 

You can also clone the repostiory and open the notebooks in [docs/src](https://github.com/marius311/CMBLensing.jl/tree/master/docs/src) if you want to run them locally (which will usually lead to higher performance). The notebooks are stored as `.md` files rather than `.ipynb` format. Its recommented to install [Jupytext](jupytext) (`pip install jupytext`) and then you can run these `.md` directly from Jupyterlab by right-clicking on them and selecting `Open With -> Notebook`. Otherwise, run the script `docs/make_notebooks.sh` to convert the `.md` files to `.ipynb` which you can then open as desired. 


## Installation

### Requirements

* Julia 1.5
* _(recommended)_ An Nvidia GPU and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for GPU support
* _(recommended)_ FFTW.jl built with [`JULIA_FFTW_PROVIDER=MKL`](https://juliamath.github.io/FFTW.jl/stable/#Installation-1) for faster CPU FFTs 
* _(recommended)_ Python 3 + matplotlib (used for plotting)
* _(recommended)_ [pycamb](https://github.com/cmbant/CAMB) to generate $C_\ell$'s
* _(optional)_ [healpy](https://github.com/healpy/healpy) for experimental curved sky support

### Native installation

To install the Julia package locally, run:

```juliapkg
pkg> add CMBLensing
```

(type `]` at the Julia REPL to reach the `pkg>` prompt)

It's recommended to link Julia to Intel MKL FFT libraries, which provide significantly faster FFTs when running on CPU. This can be done easily by running `ENV["JULIA_FFTW_PROVIDER"]="MKL"; using Pkg; Pkg.build("FFTW")` from Julia and restarting the session (see also [here](https://github.com/JuliaMath/FFTW.jl#mkl)).

### Docker installation

Also provided is a Docker container which includes a Jupyterlab server and all the recommended and optional dependencies to run and use `CMBLensing.jl`. Launch this container with:

```sh
git clone https://github.com/marius311/CMBLensing.jl.git
cd CMBLensing.jl
docker-compose pull
docker-compose up
```

The first time you run this, it will automatically download the (~1Gb) container from the Docker hub. The command will prompt you with the URL which you should open in a browser to access the notebook.

To run the notebook on a different port than the default `8888`, do `PORT=1234 docker-compose up` where `1234` is whatever port number you want.

You can also build the container locally by replacing `docker-compose pull` with `docker-compose build` above.
