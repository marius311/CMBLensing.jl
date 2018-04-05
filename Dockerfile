FROM alpine:edge

RUN echo "@testing http://nl.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories \
    && apk add --update \
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

# install CMBLensing dependencies and precompile
COPY REQUIRE $HOME/.julia/v0.6/CMBLensing/REQUIRE
RUN chown -R $NB_USER $HOME && chgrp -R $NB_USER $HOME
USER ${NB_USER}
RUN julia -e 'ENV["PYTHON"]="python3"; Pkg.add("IJulia"); for p in Pkg.available() try @eval using $(Symbol(p)); println(p); end; end'

# install CMBLensing itself (do so separately here to improve Docker caching)
COPY . $HOME/.julia/v0.6/CMBLensing
RUN julia -e "using CMBLensing, Optim"

# launch notebook
WORKDIR $HOME/.julia/v0.6/CMBLensing
CMD jupyter-notebook --ip=* --no-browser