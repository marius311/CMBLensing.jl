FROM ubuntu:20.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        curl \
        expect \
        ffmpeg \
        gfortran \
        git \
        libbz2-dev \
        libcfitsio-dev \
        libffi-dev \
        libjpeg8-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        nodejs \
        npm \
        python3 \
        python3-pip \
        python3-openssl \
        tk-dev \
        wget \
        xz-utils \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
    
## install julia
RUN mkdir /opt/julia \
    && curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.0-rc2-linux-x86_64.tar.gz | tar zxf - -C /opt/julia --strip=1 \
    && chown -R 1000 /opt/julia \
    && ln -s /opt/julia/bin/julia /usr/local/bin

## setup unprivileged user needed for mybinder.org
ARG NB_USER=cosmo
ARG NB_UID=1000
ENV USER $NB_USER
ENV NB_UID $NB_UID
ENV HOME /home/$NB_USER
ENV PATH=$HOME/.local/bin:$PATH
RUN adduser --disabled-password --gecos "Default user" --uid $NB_UID $NB_USER
USER $NB_USER

## install Python packages
# see https://github.com/jupyter/jupyter_client/issues/637 re: jupyter-client==6.1.12
RUN pip3 install --no-cache-dir \
        jinja2 \
        juliacall \
        jupyterlab \
        jupytext \
        matplotlib \
        nbconvert \
    && rm -rf $HOME/.cache

## build args
# build with PRECOMPILE=0 and/or PACKAGECOMPILE=0 to skip precompilation steps,
# which makes for a quicker build but slower startup (mostly useful for
# debugging)
ARG PRECOMPILE=1
ARG PACKAGECOMPILE=1
# JULIA_FFTW_PROVIDER="FFTW" can be used for quicker building / smaller image
# (but slower execution)
ARG JULIA_FFTW_PROVIDER=FFTW


## install CMBLensing
# to improve Docker caching, we first precompile dependencies by copying in
# Project.toml (and making a dummy src/CMBLensing.jl which we have to), so that
# other changes to files in src/ won't have to redo these steps
COPY --chown=1000 Project.toml CondaPkg.toml $HOME/CMBLensing/
COPY --chown=1000 docs/Project.toml $HOME/CMBLensing/docs/
RUN mkdir $HOME/CMBLensing/src && mkdir $HOME/CMBLensing/docs/build && touch $HOME/CMBLensing/src/CMBLensing.jl
ENV JULIA_PROJECT=$HOME/CMBLensing/docs
RUN JULIA_PKG_PRECOMPILE_AUTO=0 julia -e 'using Pkg; pkg"status; dev ~/CMBLensing; instantiate"' \
    && rm -rf $HOME/.julia/conda/3/pkgs
COPY --chown=1000 src $HOME/CMBLensing/src
COPY --chown=1000 ext $HOME/CMBLensing/ext
COPY --chown=1000 dat $HOME/CMBLensing/dat
RUN (test $PRECOMPILE = 0 || julia -e 'using Pkg; pkg"precompile"')



## PackageCompiler
# bake CMBLensing into the system image to further speed up load times and
# reduce memory usage during package load (the latter is necessary otherwise we
# hit the mybinder memory limit)
RUN test $PACKAGECOMPILE = 0 \
    || julia -e 'using PackageCompiler, Libdl; create_sysimage(["CMBLensing"], cpu_target="generic", sysimage_path=abspath(Sys.BINDIR,"..","lib","julia","sys."*Libdl.dlext))'

## execute documentation notebooks and save outputs
COPY --chown=1000 docs/src $HOME/CMBLensing/docs/src
COPY --chown=1000 docs/ipython_config.py $HOME/.ipython/profile_default/ipython_config.py
WORKDIR $HOME/CMBLensing/docs/src
ARG RUNDOCS=1

RUN jupytext --to notebook *.md \
    && rm *.md \
    && find . -not -name "*gpu*" -name "*.ipynb" \
    && test $RUNDOCS = 0 || for f in $(find . -not -name "*gpu*" -name "*.ipynb"); do \
        jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $f || ! break; \
    done

## prepare for building documentation
COPY --chown=1000 docs/make.jl docs/index.html docs/documenter.tpl $HOME/CMBLensing/docs/
COPY --chown=1000 docs/src-staging $HOME/CMBLensing/docs/src-staging
COPY --chown=1000 README.md $HOME/CMBLensing/
# shortens array output in Julia notebooks
ENV LINES=10
ARG MAKEDOCS=0
RUN test $MAKEDOCS = 0 || julia ../make.jl

## set up Jupyterlab
ENV PORT 8888
ENV SHELL=bash
CMD jupyter lab --ip=0.0.0.0 --no-browser --port $PORT
