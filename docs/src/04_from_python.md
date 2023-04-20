---
jupyter:
  jupytext:
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.10
---

```python
%pylab inline
```

# Calling from Python


You can call Julia and CMBLensing.jl directly from Python. This is made possible by the [juliacall](https://pypi.org/project/juliacall/) package. You can install it into your Python environment with, e.g.:

```shell
$ pip install --user juliacall
```


## Basics of calling Julia


Once juliacall is installed, first point it to the Julia environment environment you want active (the one with your desired CMBLensing version in it):

```python
import os
os.environ["PYTHON_JULIAPKG_PROJECT"] = "/home/cosmo/CMBLensing/docs"
os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"
```

Then import juliacall:

```python
from juliacall import Main as jl
```

The `jl` object represents the Julia `Main` model, for example:

```python
jl.cos(1) # <--- this is Julia's cosine function
```

You can also run arbitrary Julia code (this is helpful if you want to use Julia language features or syntax which don't exist in Python). For example, `1:10` is not valid Python syntax, but you can do:

```python
jl.seval("1:10")
```

## Calling CMBLensing.jl


You can use `seval` to essentially just paste Julia code into Python session, for example, following the [Lensing a flat-sky map](../01_lense_a_map/) example:

```python
jl.seval("""
using CMBLensing
""")
```

Next, we simulate some data:

```python
jl.seval("""
(;f, ϕ) = load_sim(
    θpix  = 2,
    Nside = 256,
    T     = Float32,
    pol   = :P
);
""");
```

...and we could continue the example as desired.


Variables defined by `seval` can be accessed directly in the `Main` module, and are automatically converted to Python-usage objects, e.g:

```python
matshow(jl.seval("f[:Ex]"))
```

You can also pass Python objects into Julia function, and they are converted as well:

```python
jl.FlatMap(np.random.randn(10,10))
```

See the [documentation](https://cjdoris.github.io/PythonCall.jl/stable/) for PythonCall / juliacall for more details.


## Plotting


If you want to use special plotting of maps defined in Julia, be sure to use the Julia plot function not the Python one:

```python
jl.seval("""
using PythonPlot
""")
```

```python
jl.plot(jl.f)
```
