FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y \
        curl \
        cython3 \
        gfortran \
        libcfitsio2 \
        libgsl-dev \
        python3-healpy \
        python3-numpy \
        python3-pip \
        python3-scipy \
        python3-zmq \
    && pip3 install --no-cache-dir \
        notebook==5.* \
        jupyter_contrib_nbextensions==0.3.1 \ 
        matplotlib==2.* \
        setuptools==20.4 \
    && jupyter contrib nbextension install \
    && jupyter nbextension enable toc2/main --system \
    && rm -rf /var/lib/apt/lists/*

# install julia
RUN mkdir /opt/julia \
    && curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.3-linux-x86_64.tar.gz | tar zxf - -C /opt/julia --strip=1 \
    && ln -s /opt/julia/bin/julia /usr/local/bin

# setup unprivileged user needed for mybinder.org
ENV NB_USER cosmo
ENV NB_UID 1000
ENV HOME /home/${NB_USER}
RUN adduser --disabled-password --gecos "cosmo" --uid ${NB_UID} ${NB_USER}
USER cosmo

# install CAMB
RUN mkdir -p $HOME/src/camb \
    && curl -L https://github.com/cmbant/camb/tarball/6fc83ba | tar zxf - -C $HOME/src/camb --strip=1 \
    && cd $HOME/src/camb/pycamb \
    && python3 setup.py install --user

# launch notebook
WORKDIR $HOME
COPY --chown=1000 . $HOME/CMBLensing.jl

RUN PYTHON=python3 JULIA_FFTW_PROVIDER=MKL julia -e 'using Pkg; try; Pkg.REPLMode.pkgstr("dev CMBLensing.jl"); catch; end; Pkg.REPLMode.pkgstr("dev CMBLensing.jl; add IJulia; build; precompile")'

RUN mkdir -p $HOME/.julia/config && mv $HOME/CMBLensing.jl/startup.jl $HOME/.julia/config/

CMD jupyter-notebook --ip=0.0.0.0 --no-browser
