---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Calling from Python


Calling Julia and CMBLensing.jl directly from Python is very transparent. This is made possible by the [PyJulia](https://pyjulia.readthedocs.io/en/latest/index.html) package. You can install it into your Python environment with, e.g.:

```shell
$ pip install --user julia
```


**Important:** If your Python executable is statically-linked (this is quite often the case, e.g. its the default on Ubuntu and Conda) you need one extra step. Basically, instead of running `python` or `ipython` at the command line to launch your interpreter, run `python-jl`  or `python-jl -m IPython`, respectively. If you use Jupyter, you'll need to edit your `kernel.json` file (you can find its location via `jupyter kernelspec list`) and change it to use `python-jl`.

The wrapper script `python-jl` does some special initializion but otherwise drops you into the Python/IPython interpreter that you are familiar with. 

The [PyJulia docs](https://pyjulia.readthedocs.io/en/latest/troubleshooting.html#your-python-interpreter-is-statically-linked-to-libpython) also give instructions on how to install a dynamically-linked Python executable which is the most ideal solution, and only slightly more work than above.


## Basics of calling Julia


Once PyJulia is installed, you can access any Julia package `Foo` from the Python package `julia.Foo`, and everything pretty much works the same.

```python
import julia.Base
```

```python
julia.Base.cos(1) # <--- this is Julia's cosine function
```

You can also run arbitrary Julia code with the `%julia` cell magic (this is helpful if you want to use Julia language features or syntax which don't exist in Python):

```python
%load_ext julia.magic
```

For example, `1:10` is not valid Python syntax, but we can do:

```python
%julia 1:10
```

The cell magic lets you interpolate values from Python into the Julia expression, which can be a convenient way to pass values back and forth:

```python
x = %julia 1 + 2
```

```python
%julia 2 * $x
```

## Calling CMBLensing.jl


### Via magic


The most robust way to call CMBLensing.jl from Python is just to wrap everything in Julia magic and interpolate things back and forth as-needed. Lets try and follow the [Lensing a flat-sky map](../01_lense_a_map/) example from Python. First, we load the package:

```julia
using CMBLensing
```

Next, we simulate some data:

```julia
@unpack f,ϕ = load_sim(
    θpix  = 2,
    Nside = 256,
    T     = Float32,
    pol   = :I
);
```

Similarly, the rest of the commands from that example will work in Python if just called via Julia magic.


At any point, you can do whatever you'd like with any of the results stored in Julia variables, e.g. transferring the simulated maps back as Python arrays,

```python
f = %julia f[:Ix]
f
```

You can also pass variables back to Julia, e.g.

```python
%julia g = FlatMap($f);
```

### Directly


You can also call Julia directly without magic, which sometimes offers more flexibility, although has some limitations. 

To do so, first import CMBLensing. into Python. In Julia, `using CMBLensing` imports all of the CMBLensing symbols into the current namespace. In Python this is:

```python
from julia.CMBLensing import *
```

If we want to call `load_sim` as before, we must take into account a few things:

* You won't be able to use the `@unpack` macro since macros on arbitrary code don't exist in Python.
* `Float32` isn't imported into Python by default, so you'll need to specify the module. 
* The `:P` is invalid syntax in Python, you should use a string `"P"` instead. 

Given all of that, the call will look like:

```python
sim = load_sim(
    θpix  = 2, 
    Nside = 256, 
    T     = julia.Base.Float32, 
    pol   = "P"
)
```

If we wish to grab the lensing potential from the result, there's an additional consideration. Python does not differentiate between the characters `ϕ (\phi)` and `φ (\varphi)`, and maps both of them back to `φ (\varphi)` in Julia, which unfortunately is the wrong one for CMBLensing (which instead makes extensive use of the variable name `ϕ (\phi)`). Thus, calling `sim.ϕ` from Python does not work. Instead, we have to do that part in Julia:

```python
ϕ = %julia $sim.ϕ
```

## Plotting


To plot, we need to use the plot function from Julia's PyPlot, since this will know about plotting CMBLensing objects. 

```python
from julia.PyPlot import plot
```

```python
%matplotlib inline
```

```python
plot(ϕ);
```

For non-CMBLensing objects, this plot function will just pass-through to matplotlib, so will not affect affect your session otherwise.

```python
plot([1,2,3]);
```
