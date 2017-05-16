FROM ubuntu:xenial

WORKDIR /root

RUN apt-get update && apt-get install -y curl cython gcc hdf5-tools make python-numpy python-matplotlib \
    && mkdir julia \
    && curl -L https://julialang.s3.amazonaws.com/bin/linux/x64/0.6/julia-0.6-latest-linux-x86_64.tar.gz | tar zxf - -C julia --strip=1 \
    && ln -s /root/julia/bin/julia /usr/bin/ \
    && mkdir class \
    && curl -L https://github.com/lesgourg/class_public/tarball/992b18b | tar zxf - -C class --strip=1 \
    && cd class \
    && sed -i 's/CCFLAG = -g -fPIC/CCFLAG = -g -fPIC -fno-tree-vectorize/g' Makefile \
    && make all \
    && rm -rf /var/lib/apt/lists/*
    
# precompile
COPY REQUIRE /root/baylens/REQUIRE
RUN julia -e "Pkg.init()" \
    && ln -s $HOME/baylens /root/.julia/v0.6/CMBLensing \
    && julia -e "Pkg.resolve(); Pkg.checkout(\"ODE\"); for p in Pkg.available() try eval(:(using \$(Symbol(p)))) end end"
    
COPY . /root/baylens
ENV JULIA_NUM_THREADS=1
WORKDIR /root/baylens/scripts
