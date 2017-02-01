<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Design Docs](#design-docs)
	- [Type hierarchy](#type-hierarchy)
		- [Note on differences with Ethan's prototype](#note-on-differences-with-ethans-prototype)
		- [Type naming convention](#type-naming-convention)
		- [Type parameters naming convention](#type-parameters-naming-convention)
	- [Generated Planned FFT](#generated-planned-fft)
	- [Handling the underlying data](#handling-the-underlying-data)
	- [`getindex` convenience functions](#getindex-convenience-functions)
	- [Interface](#interface)
		- [Field](#field)

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

The meaning of the basis `B` for an operator is that, by default, fields will automatically be converted to that basis before being fed into the operator, therefore you only need to define the action of an operator in one basis. Of course, if you have a more efficient way to act with the operator in multiple bases, you could just add methods working in those bases which would supercede the default behavior. 


Intended possibilities are,

* `Pix` 
    * `Flat`
    * `Healpix`
* `Spin` 
    * `S0` (spin 0, i.e. T or ϕ)
    * `S2` (spin 2, i.e. Q&U)
    * `S02` (i.e. a spin0 and a spin2 component, so T/Q/U)
* `Basis`
    * For `S0`, we have `Map` (real space) and`Fourier` (Fourier space)
    * For `S2` we have `EBMap`, `EBFourier`, `QUMap`, and `QUFourier`. 
	
	
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
    
	but I only need the `promote_rule`. 
    
    ~~A consequence of my using the abstract types is that I can't use `promote_rules` to handle arithmetic like you do. This is basically because there's no way to do something like,~~

	```julia
	promote_rule{P,S,B1,B2,F1<:Field{P,S,B1}, F2<:Field{P,S,B2}}(::Type{F1}, Type{F2}) = ...
	```

	~~I do like your way more since its more "Julian", but at least like you mention in your code comment, this should be possible in 0.6. For now I do things by hand with `@swappable`, and we could switch to `promote_rules` once 0.6 comes out with minimal impact elsewhere.~~
	
	**Update**: Actually, I've got a way which uses `promote` working in 0.5 which works by adding `promote_type` methods for Field types, rather than `promote_rule`. Its very slightly hacky, and can be made entirely not-hacky on 0.6. 
	
	
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

## Converting bases

The bases types like `Map` or `QUFourier` are used not just to specify the type but also to convert bases. E.g., `Map(x)` where `x::FlatS0Fourier` would return a `FlatS0Map`. 

## Generated Planned FFT

For flat sky pixelization (and possibly others in the future), we carry around information about the number of pixels and pixel area in the type via,

```julia
abstract Flat{Θpix,Nside} <: Pix
```

These parameters, along with the elementy type of the matrix (like Float64), uniquely define an FFTgrid object, which has in it info about the FFT (like Δx and Δy) and the FFT operator itself. You can get an FFTgrid via `FFTgrid(T,P)` where T is the matrix type and P is a `Flat` type. 

Computing the FFT of a matrix `x` would thus look like `FFTgrid(T,P).FFT * x`. As an additional convenience, one can use instead `ℱ(P)*x`, which infers the `T` type automatically and doesn't require typing the `.FFT` field. 

Note `FFTgrid(T,P)` only does the precomputation the first time its called, so its very fast after that. 

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

## `getindex` convenience functions

We give a convenient way to access the underlying data when needed. Accessing the data using the method described below adds some overhead as compared to just performing algebra on the Field objects, so it may not be suited for speed-critical sections of code, but can be quite handy for interactive work. 

Essentially, one can use `[]` to get a desired data field, and if the Field object needs to be converted it will be done automatically. That is, if we have two types,

```julia
immutable FlatS0Map{T<:Real,P<:Flat} <: Field{P,S0,Map}
    Tx::Matrix{T}
end

immutable FlatS0Fourier{T<:Real,P<:Flat} <: Field{P,S0,Fourier}
    Tl::Matrix{Complex{T}}
end
```

and an object of type `x::FlatS0Map`, then `x[:Tx]` gets the `Tx` field, and `x[:Tl]` *converts* it to a Fourier basis `FlatS0Fourier` object and then gets the `:Tl`. 

This is implemented in our custom `getindex(f::Field,x::Symbol)
` function, which works by

1. Finding all the defined types which share the same supertype as `f`
2. Searching through them for one with a field `x`
3. `convert`-ing `f` to that type (the `convert` function for this obviously needs to be defined)
4. Getting the field



## Interface

### Field

To create a new `Field` type, first create a set of types with different bases, 

```julia
type MyFieldMap <: Field{P,S,Map}
	# ...
end

type MyFieldFourier <: Field{P,S,Fourier}
	# ...
end
```
(where `P` and `S` are replaced by whatever `Pix` and `Spin` types your field is implementing).

Then you need to define,

1. Methods for how to transform these two fields between bases, 

	```julia
	Fourier(f::MyFieldMap) = ...  # should return a MyFieldFourier
	Map(f::MyFieldFourier) = ...  # should return a MyFieldMap
	```

	(the bases types which appear here must be the same ones as in the type definition)
	
2. A set of rules for which bases the result is in when you operate on mixed-bases objects via, 

	```julia
	@swappable promote_type(::Type{FlatS0Map}, ::Type{FlatS0Fourier}) = FlatS0Map
	```

	In this case we've said that e.g. Map+Fourier results in a Map. If instead two bases you have N bases, you would need N(N-1)/2 such definitions to specify all possible cases. 


With these definitions, all arithmetic with these fields will work, as will the `getindex` functionality.

If you want to be able to use to automatic vector to/from conversion, you also need to define,

```julia
""" Return vector representation """
tovec(f::F) = ... # should return an AbstractVector

""" Convert vector representation back to Field """
fromvec{T<:F}(::Type{T}, vec::AbstractVector) = ... # should return a type T
```
for `F` of both `MyFieldMap` and `MyFieldFourier` (`Union` might handly to combine function definitions). 
