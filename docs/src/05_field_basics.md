---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.4
  kernelspec:
    display_name: Julia 1.7.0
    language: julia
    name: julia-1.7
---

# Field Basics

```julia
using CMBLensing
```

## Base Fields

The basic building blocks of CMBLensing.jl are CMB "fields", like temperature, Q or U polarization, or the lensing potential $\phi$. These types are all encompassed by the abstract type `Field`, with some concrete examples including `FlatMap` for a flat-sky map projection, or `FlatQUMap` for Q/U polarization, etc...


`Flat` fields are just thin wrappers around Julia arrays, e.g.

```julia
Ix = rand(2,2)
```

```julia
f = FlatMap(Ix)
```

When displayed, you can see the pixels in the 2x2 map have been splayed out into a length-4 array. This is intentional, as even though the maps themselves are two-dimensional, it is extremely useful conceptually to think of fields as vectors (which they are, in fact, as they form an [abstract vector space](https://en.wikipedia.org/wiki/Vector_space)). This tie to vector spaces is deeply rooted in CMBLensing, to the extent that `Field` objects are a subtype of Julia's own `AbstractVector` type, 

```julia
f isa AbstractVector
```

The data itself, however, is still stored as the original 2x2 matrix, and can be accessed as follows,

```julia
f.Ix
```

But since `Fields` are vectors, they can be tranposed,

```julia
f'
```

inner products can be computed,

```julia
f' * f
```

and they can be added with each other as well as multiplied by scalars,

```julia
2*f+f
```

## Diagonal operators


Vector spaces have linear operators which act on the vectors. Linear operators correpsond to matrices, thus for a map with $N$ total pixels, a general linear operator would be an $N$-by-$N$ matrix, which for even modest map sizes becomes far too large to actually store. Thus, an important class of linear operators are ones which are diagonal, since these can actually be stored. CMBLensing uses Julia's builtin `Diagonal` to represent these. `Diagonal(f)` takes a vector `f` and puts it on the diagonal of the matrix:

```julia
Diagonal(f)
```

Multiplying this operator by the original map is then a matrix-vector product:

```julia
Diagonal(f) * f
```

Note that this is also equal to the the pointwise multiplication of `f` with itself:

```julia
f .* f
```

## Field Tuples


You can put `Fields` together into tuples. For example, 

```julia
a = FlatMap(rand(2,2))
b = FlatMap(rand(2,2));
```

```julia
FieldTuple(a,b)
```

The components can also have names:

```julia
ft = FieldTuple(a=a, b=b)
```

which can be accessed later:

```julia
ft.a
```

`FieldTuples` have all of the same behavior of individual fields. Indeed, spin fields like QU or IQU are simply special `FieldTuples`:

```julia
fqu = FlatQUMap(a,b)
fqu isa FieldTuple
```

## Field Vectors


*in progress*


## Basis Conversion


All fields are tagged as to which basis they are stored in. You can convert them to other bases by calling the basis type on them:

```julia
f
```

```julia
f′ = Fourier(f)
```

Basis conversion is usually done automatically for you. E.g. here `f′` is automatically converted to a `FlatMap` before addition:

```julia
f + f′
```

A key feature of `Diagonal` operators is they convert the field they are acting on to the right basis before multiplication:

```julia
Diagonal(f) * f′
```

A `FlatMap` times a `FlatFourier` doesn't have a natural linear algebra meaning so its an error:

```julia tags=["raises-exception"]
f * f′
```

## Properties and indices


`FlatMap` and `FlatFourier` can be indexed directly like arrays. If given 1D indices, this is the index into the vector representation:

```julia
f
```

```julia
f[1], f[2], f[3], f[4]
```

```julia tags=["raises-exception"]
f[5]
```

Or with a 2D index, this indexes directly into the 2D map:

```julia
f[1,1], f[2,1], f[1,2], f[2,2]
```

*Note:* there is no overhead to indexing `f` in this way as compared to working directly on the underlying array.


For other fields which are built on `FieldTuples`, 1D indexing will instead index the tuple indices:

```julia
ft
```

```julia
ft[1]
```

```julia
ft[2]
```

```julia tags=["raises-exception"]
ft[3]
```

To get the underlying data arrays, use the object's properties:

```julia
f.Ix
```

You can always find out what properties are available by typing `f.<Tab>`. For example, if you typed `ft` then hit `<Tab>` you'd get:

```julia
ft |> propertynames
```

For a `FieldTuple` like the `FlatQUMap` object, `fqu`, you can get each individual Q or U field:

```julia
fqu.Q
```

Or `fqu.Qx` which is shorthand for `fqu.Q.Ix`:

```julia
fqu.Q.Ix === fqu.Qx
```

If you convert `f` to Fourier space, it would have the `Il` property to get the Fourier coefficients of the $I$ component:

```julia
Fourier(f).Il
```

For convenience, you can index fields with brackets `[]` and any necessary conversions will be done automatically:

```julia
f[:Il]
```

This works between any bases. For example. `fqu` is originally `QUMap` but we can convert to `EBFourier` and get the `El` coefficients:

```julia
fqu[:El]
```

The general rule to keep in mind for these two ways of accessing the underlying data is:

* **Properties** (i.e. `f.Ix`) are type-stable and get you the underlying data arrays, even recursively from special `FieldTuples` like `FlatQUMap`, etc... If these arrays are modified, they affect the original field.
* **Indices** (i.e. `f[:Ix]`) are not type-stable, and may or may not be one of the underlying data arrays (because a basis conversion may have been performed). They should be used for getting (not setting) data, and in non-performance-critical code. 
