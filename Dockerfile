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
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        nodejs \
        npm \
        python-openssl \
        tk-dev \
        wget \
        xz-utils \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
    
## install julia
RUN mkdir /opt/julia \
    && curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.0-linux-x86_64.tar.gz | tar zxf - -C /opt/julia --strip=1 \
    && chown -R 1000 /opt/julia \
    && ln -s /opt/julia/bin/julia /usr/local/bin

## setup unprivileged user needed for mybinder.org
ARG NB_USER=cosmo
ARG NB_UID=1000
ENV USER $NB_USER
ENV NB_UID $NB_UID
ENV HOME /home/$NB_USER
RUN adduser --disabled-password --gecos "Default user" --uid $NB_UID $NB_USER
USER $NB_USER

## pyenv
# install python with pyenv since we need a dynamically-linked executable so
# that PyJulia works
ENV PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH"
RUN curl https://pyenv.run | bash \
    && CFLAGS="-O2" PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.7.3 \
    && pyenv global 3.7.3

## install Python packages
# see https://github.com/jupyter/jupyter_client/issues/637 re: jupyter-client==6.1.12
RUN pip install --no-cache-dir \
        cython \
        julia \
        "jupyterlab>=3" \
        jupyter-client==6.1.12 \
        jupytext \
        matplotlib \
        "nbconvert<6" \
        numpy \
        scipy \
        setuptools \
    && rm -rf $HOME/.cache

## install CAMB
RUN mkdir -p $HOME/src/camb \
    && curl -L https://github.com/cmbant/camb/tarball/21a56ef | tar zxf - -C $HOME/src/camb --strip=1 \
    && cd $HOME/src/camb \
    && python setup.py make install


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
COPY --chown=1000 Project.toml $HOME/CMBLensing/
COPY --chown=1000 docs/Project.toml $HOME/CMBLensing/docs/
RUN mkdir $HOME/CMBLensing/src && touch $HOME/CMBLensing/src/CMBLensing.jl
ENV JULIA_PROJECT=$HOME/CMBLensing/docs
RUN JULIA_PKG_PRECOMPILE_AUTO=0 julia -e 'using Pkg; pkg"dev ~/CMBLensing; instantiate"' \
    && rm -rf $HOME/.julia/conda/3/pkgs
COPY --chown=1000 src $HOME/CMBLensing/src
RUN (test $PRECOMPILE = 0 || julia -e 'using Pkg; pkg"precompile"')



## PackageCompiler
# bake CMBLensing into the system image to further speed up load times and
# reduce memory usage during package load (the latter is necessary otherwise we
# hit the mybinder memory limit)
RUN test $PACKAGECOMPILE = 0 \
    || julia -e 'using PackageCompiler; create_sysimage([:CMBLensing],cpu_target="generic",replace_default=true)'

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

## set up Jupyterlab
ENV PORT 8888
ENV SHELL=bash
CMD jupyter lab --ip=0.0.0.0 --no-browser --port $PORT
