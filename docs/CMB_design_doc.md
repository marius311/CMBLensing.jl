<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Design Docs](#design-docs)
	- [Type hierarchy](#type-hierarchy)
		- [Note on differences with Ethan's prototype](#note-on-differences-with-ethans-prototype)
		- [Type naming convention](#type-naming-convention)
		- [Type parameters naming convention](#type-parameters-naming-convention)
	- [Generated Planned FFT](#generated-planned-fft)
	- [Handling the underlying data](#handling-the-underlying-data)

<!-- /TOC -->

# Design Docs

This document contains some explanation for the design choices made in CMBFields.jl. 

The main goals of CMBFields.jl are, 

1. Provide a convenient way to manipulate CMB fields (like T,E,B,ϕ,..) and linear operators on those fields where the user never has to worry about what basis something is represented in and can perform linear algebra on fields and linear operators using normal notation.
2. Allow using external library algorithms (e.g. conjugate gradient, etc...) with minimal boiler plate code to allow interacting with our types.
3. Provide a code framework which is forward thinking and allows one to add in new types of fields with minimal code (e.g. will allow easy addition of Healpix fields, etc...)
4. Be fast. 

## Type hierarchy

We assume a field object's behavior is described by a choice of (pixelization, spin, basis), hence the definition for a field,

```julia
abstract Field{P<:Pix, S<:Spin, B<:Basis}
```

and for an operator that can act on a field,

```julia
abstract LinearFieldOp{P<:Pix, S<:Spin, B<:Basis}
```

Intended possibilities are,

* `Pix` 
    * `Flat`
    * `Healpix`
* `Spin` 
    * `S0` (spin 0, i.e. T or ϕ)
    * `S2` (spin 2, i.e. Q&U)
    * `S02` (i.e. a spin0 and a spin2 component, so T/Q/U)
* `Basis`
    * `Map` (real space)
    * `Fourier` (Fourier space)
    * *TODO: Q&U and E&B should be different bases for `spin2`*

although the framework should be general enough to allow adding more we haven't thought of.

As much as possible of the behavior is defined in terms of this abstract type, so concrete types implementing this inherit as much behavior as possible and require minimal amount of extra code.

### Note on differences with Ethan's prototype

For you (Ethan), each set of (pixelization, spin, basis) is a concrete type (eg `Spin0Pix`), whereas for me, these are abstract types (eg `Field{Flat,S0,Map}`). Then I have concrete types which implement each of these abstract types (eg `FlatS0Map`). The reason I think this is better is two things,

1. I can define more behavior in terms of the abstract types which leads to less code whenever you add any new type (including possibly ones we haven't thought of yet) For example, say I want to add a new field type `NewType`. Besides defining how to (inv/)Fourier transform it (which for you happens in `convert` and for me in the `*` operator), you need the following,

	```julia
	Freq{p,n}(s::NewTypePix{p,n})  = convert(NewTypeFreq{p,n}, s)
	Freq{p,n}(s::NewTypeFreq{p,n}) = s
	Pix{p,n}(s::NewTypeFreq{p,n}) = convert(NewTypePix{p,n}, s)
	Pix{p,n}(s::NewTypePix{p,n})  = s
	promote_rule{p,n}(::Type{NewTypePix{p,n}}, ::Type{NewTypeFreq{p,n}})  = NewTypePix{p,n}
	```
    
	but I need nothing extra. 
    
    A consequence of my using the abstract types is that I can't use `promote_rules` to handle arithmetic like you do. This is basically because there's no way to do something like,

	```julia
	promote_rule{P,S,B1,B2,F1<:Field{P,S,B1}, F2<:Field{P,S,B2}}(::Type{F1}, Type{F2}) = ...
	```

	I do like your way more since its more "Julian", but at least like you mention in your code comment, this should be possible in 0.6. For now I do things by hand with `@swappable`, and we could switch to `promote_rules` once 0.6 comes out with minimal impact elsewhere. 
	
2. The second reason is that its more flexible. For example, one might imagine two implementations of `Field{Flat,S0,Map}`, the default one I currently have coded up, and maybe one where we use the GPU, or ArrayFire. In that case, the new type (say `GPUFlatS0Map`) still inherits from `Field{Flat,S0,Map}`, so it gets all of the behavior from that, then maybe I have one or two functions to change. With your option, `GPUSpin0Pix` is completely independent of `Spin0Pix` and a bunch of the code needs to be copied.

One of your concerns was remembering names of things. Hopefully the naming conventions below help that (remembering `FlatS0Map` should in theory be just as easy to remember as `Field{Flat,S0,Map}`).


### Type naming convention

The default concrete types implementing `Field` should be named by concatenating their `Pix`, `Spin`, and `Basis`, in that order. For example, the default concrete type for `Flat`, `S0`, and `Map` is called `FlatS0Map`. 

If we add different implementations for the same concrete type, then the default name should still appear in its entirely somewhere. E.g., if we add a flat spin-0 map for which operations are performed on the GPU, that should be something like `GPUFlatS0Map`. 

Similarly, the entire name should appear in its entirely for operators too. A diagonal covariance of flat spin-0 map is `FlatS0MapDiagCov`.

If operators only make sense implemented in one particular basis, the basis part can be omitted. E.g., it makes sense to store covariances in map and fourier basis, hence `FlatS0MapDiagCov` and `FlatS0FourierDiagCov`, however it doesn't make sense to code up the lensing operator both in map and fourier space, hence we just have `FlatS0LensingOp` (which we have chosen to implement assuming the input is in map space; of course `FlatS0LensingOp` can be applied to a `Fourier` object, which will be converted automatically). 

### Type parameters naming convention

If the `Pix`, `Spin`, and `Basis` parameters appear in a parametric function definition, they should be called `P`, `S`, and `B`, i.e.

```julia
foo{P,S,B}(f::Field{P,S,B}) = ...
```

## Generated Planned FFT

For flat sky pixelization (and possibly others in the future), we carry around information about the number of pixels and pixel area in the type via,

```julia
abstract Flat{Θpix,Nside} <: Pix
```

Then the `ℱ` function acting on a `Flat` type gets us the planned FFT,

```julia
ℱ(Flat{1,1024}) # <-- returns an FFT object
```

This allows us to make `ℱ` a `@generated` function so its only ever called once for a given `{Θpix,Nside}`.

## Handling the underlying data

Different field types store their underlying data in one or more matrices or vectors.

* In my (Marius) implementation, this can be stored in any fashion, then the `data` function can be used to specify which fields are data, 

    ```julia

    # one way to store Q&U
    type QUField1{T}
        Q::Matrix{T}
        U::Matrix{T}
    end
    data(f::QUField1) = (f.Q, f.U)

    # another way to store Q&U
    type QUField2{T}
        d::Array{T,3} # [:,:,1] is Q and [:,:,2] is U 
    end
    data(f::QUField1) = (f.d)
    ```

    (note the expclit definitions of `data` above are not necessary, since the default implementation of `data` takes all of the fields)

    The significance of "data" is that its what is used by default to broadcast operations, 

    ```julia
    +{T<:Field}(a::T, b::T) = T(map(+,map(data,(a,b))...)..., meta(a)...)
    ```

    Additionally, `Field` objects can specify which fields are `meta` (i.e. metadata which doesn't participate in operations). 


* In Ethan's implementation, the data is always just assumed to be in the `d` field, 

    ```julia
    # only this way will work:
    type QUField2{T}
        d::Array{T,3} # [:,:,1] is Q and [:,:,2] is U 
    end
    ```
    
    which is then used,
    
    ```julia
    +{B<:BasisCoeff}(c1::B, c2::B) =  B( +(c1.d, c2.d) )
    ```

The two options are really not much different, but mine (Marius) is slightly more flexible since it lets you lay out the data however you'd like (reasons to do so might be e.g. so that you don't have remember the index of Q/U into an array). Mine doesn't actually lead to needing more boilerplate code since I provide a default `data()` which just assumes everything is data. Thus I'd favor my way, but I don't feel too strongly about it.
