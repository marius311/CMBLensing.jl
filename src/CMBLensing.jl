module CMBLensing

using Base.Broadcast: AbstractArrayStyle, ArrayStyle, Broadcasted, broadcasted, DefaultArrayStyle, flatten, preprocess_args, Style
using Base.Iterators: repeated
using Base.Threads
using Base: @propagate_inbounds, show_vector, show_default, showarg, show_datatype, typed_vcat
using Combinatorics
using DataStructures
using Distributed
using FFTW
# using Images: feature_transform, imfilter
# using Images.Kernel
using InteractiveUtils
using Interpolations
using JLD2
using FileIO
using Loess
using LinearAlgebra
using MacroTools: @capture, combinedef, isexpr, postwalk, splitdef
using Match
using Markdown
using Memoize
using Optim: optimize
using Parameters
using Printf
using ProgressMeter
using PyCall
using PyPlot
using QuadGK
using Random
using Random: seed!
using Roots
using Requires
using Setfield
using StaticArrays: @SMatrix, @SVector, SMatrix, StaticArray, StaticArrayStyle, StaticMatrix, StaticVector, SVector
using Statistics
using StatsBase
using Strided


import Base: +, -, *, \, /, ^, ~, ≈,
    adjoint, broadcast, broadcastable, BroadcastStyle, conj, convert, copy,
    copyto!, eltype, getindex, getproperty, hcat, hvcat, inv, iterate, keys,
    length, literal_pow, materialize!, materialize, one, print_array, promote,
    promote_rule, promote_rule, promote_type, propertynames, real, setindex!,
    show, show_datatype, show_vector, similar, size, sqrt, sqrt, string,
    summary, transpose, zero
import Base.Broadcast: instantiate, preprocess
import LinearAlgebra: dot, isnan, ldiv!, logdet, mul!, pinv,
    StructuredMatrixStyle, structured_broadcast_alloc
import PyPlot: loglog, plot, semilogx, semilogy



export
    @animate, @repeated, @unpack, azeqproj, BandPassOp, cache, CachedLenseFlow,
    camb, cg, class, cov_to_Cℓ, Cℓ_2D, Cℓ_to_cov, DataSet, DerivBasis, Diagonal,
    dot, EBFourier, EBMap, FFTgrid, Field, FieldArray, FieldMatrix, FieldOrOpArray,
    FieldOrOpMatrix, FieldOrOpRowVector, FieldOrOpVector, FieldRowVector,
    FieldTuple, FieldVector, FieldVector, Flat, FlatEB, FlatEBFourier, FlatEBMap,
    FlatFourier, FlatIQUMap, FlatIQUMap, FlatMap, FlatQU, FlatQUFourier, FlatQUMap,
    FlatS0, FlatS2, FlatS2Fourier, FlatS2Map, FlatTEBFourier, Fourier, fourier∂,
    FuncOp, FΦTuple, get_Cℓ, get_Cℓ, get_Dℓ, get_αℓⁿCℓ, get_ρℓ, get_ℓ⁴Cℓ, gradhess,
    GradientCache, HealpixCap, HealpixS0Cap, HealpixS2Cap, HighPass, IdentityOp,
    InterpolatedCℓs, IsotropicHarmonicCov, LenseBasis, LenseFlow, LenseOp,
    lensing_wiener_filter, LinDiagOp, LinOp, lnP, load_healpix_sim_dataset,
    load_sim_dataset, LowPass, Map, MAP_joint, MAP_marg, map∂, MidPass, nan2zero,
    noisecls, OuterProdOp, pack, ParamDependentOp, pixwin, plot, PowerLens,
    quadratic_estimate, QUFourier, QUMap, resimulate, S0, S02, S2, sample_joint,
    shortname, simulate, sptlike_mask, symplectic_integrate, Taylens, toCℓ,
    toDℓ, tuple_adjoint, ud_grade, Ð, Ł, δf̃ϕ_δfϕ, δfϕ_δf̃ϕ, δlnP_δfϕₜ, ℓ², ℓ⁴,
    ∇, ∇², ∇¹, ∇ᵢ, ∇⁰, ∇ⁱ, ∇₀, ∇₁, ⋅, ⨳

include("util.jl")
include("numerical_algorithms.jl")
include("rfftvectors.jl")
include("generic.jl")
include("cls.jl")
include("field_tuples.jl")
include("field_vectors.jl")
include("specialops.jl")
include("algebra.jl")
include("lensing.jl")
include("flat.jl")
# include("healpix.jl")
# include("taylens.jl")
# include("vec_conv.jl")
include("plotting.jl")
# include("likelihood.jl")
# include("sampling.jl")
# include("masking.jl")
# include("quadratic_estimate.jl")
# include("properties.jl")
# include("weave_pyplot.jl")

end
