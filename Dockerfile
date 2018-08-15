FROM alpine:3.7

RUN echo "@testing http://nl.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories \
    && apk add --update \
        cmake \
        curl \
        freetype-dev \
        g++ \
        gfortran \
        hdf5@testing \
        julia \
        libpng-dev \
        make \
        mbedtls \
        musl-dev \
        openblas-dev \
        perl \
        py-numpy-dev \
        py3-numpy-f2py \
        py3-six \
        py3-zmq \
        python3 \
    && pip3 install --no-cache-dir notebook==5.4.0 tornado==4.5.3 jupyter_contrib_nbextensions==0.3.1 matplotlib \
    && jupyter contrib nbextension install \
    && jupyter nbextension enable toc2/main --system

# install CAMB
RUN mkdir -p /root/camb \
    && curl -L https://github.com/marius311/camb/tarball/scipy_optional | tar zxf - -C /root/camb --strip-components=1 \
    && sed -i -e "s/-fopenmp//g" /root/camb/Makefile \
    && cd /root/camb/pycamb \
    && python3 setup.py install

# setup unprivileged user needed for mybinder.org
ENV NB_USER bayes
ENV NB_UID 1000
ENV HOME /home/${NB_USER}
RUN adduser -Du ${NB_UID} ${NB_USER}

# add CMBLensing dependencies
COPY REQUIRE $HOME/.julia/v0.6/CMBLensing/REQUIRE
# add dependencies for Jupyter for this Docker container
RUN (echo "IJulia"; echo "ZMQ 0.5.0 0.6.0";  echo "Blosc 0.3.0 0.4.0") >> $HOME/.julia/v0.6/CMBLensing/REQUIRE
RUN chown -R $NB_USER $HOME && chgrp -R $NB_USER $HOME
USER ${NB_USER}
# install and precompile everything
RUN julia -e 'ENV["PYTHON"]="python3"; Pkg.resolve(); for p in Pkg.available() try @eval using $(Symbol(p)); println(p); end; end'

# install CMBLensing itself (do so separately here to improve Docker caching)
COPY . $HOME/.julia/v0.6/CMBLensing
RUN julia -e "using CMBLensing, Optim"

# launch notebook
WORKDIR $HOME/.julia/v0.6/CMBLensing
CMD jupyter-notebook --ip=* --no-browser
