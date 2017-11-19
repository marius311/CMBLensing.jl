# CMBLensing.jl

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/marius311/CMBLensing.jl/master?filepath=nb%2Fjointmax_128example%2Fjointmax_128example.ipynb)


This repository contains tools written in the [Julia](https://julialang.org/) language to analyze the gravitationally lensed Cosmic Microwave Background, and includes an implementation of the `LenseFlow`  algorithm as described in [Millea, Anderes, & Wandelt (2017)](https://arxiv.org/abs/1708.06753). 

The fastest way to get started with this code is to take a look at our Jupyter notebook demonstrating a 128x128 pixel example, which can launch with one click here: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/marius311/CMBLensing.jl/master?filepath=nb%2Fjointmax_128example%2Fjointmax_128example.ipynb)

This opens a fully functioning notebook in your browser, running remotely (so you do not need anything installed on your own computer), where you can execute our example code or play with anything as-you-wish. 


## Installation

Eventually, you'll probably want to install the Julia package locally, which you can do with:

```julia
julia> Pkg.clone("https://github.com/marius311/CMBLensing.jl")
```

### Other installation options

Also provided is a Docker container which includes a Jupyter notebook server and all the dependencies to run and use `CMBLensing.jl`. One advantage of using this is that you don't need any dependencies installed on your system (you just need Docker). To launch the Docker container, run the following command:

```sh
docker run --rm --name cmblensing -itp 8888 marius311/cmblensing.jl | (sleep 1 && sed -e "s/localhost/$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' cmblensing)/g")
```

The first time you run this, it will automatically download the (~300Mb) container from the Docker hub.

You can also build the container locally with:

```sh
git clone https://github.com/marius311/CMBLensing.jl
cd CMBLensing.jl
make build start
```

One advantage of this is that the `make start` command mounts the CMBLensing.jl source code as a volume inside the container, so if you edit files locally this will be immediately reflected inside the container running the notebooks. 