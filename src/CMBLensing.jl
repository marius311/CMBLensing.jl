module CMBLensing

using Base.Broadcast: AbstractArrayStyle, ArrayStyle, Broadcasted, broadcasted,
    DefaultArrayStyle, flatten, preprocess_args, Style
using Base.Iterators: repeated, product
using Base.Threads
using Base: @kwdef, @propagate_inbounds, Bottom, OneTo, showarg, show_datatype,
    show_default, show_vector, typed_vcat
using Combinatorics
using DataStructures
using Distributed
using FFTW
using InteractiveUtils
using Interpolations
using KahanSummation
using Loess
using LinearAlgebra
using LinearAlgebra: diagzero
using MacroTools: @capture, combinedef, isdef, isexpr, postwalk, splitdef
using Match
using Markdown
using Measurements
using Memoize
using Parameters
using Printf
using ProgressMeter
using PyCall
using Random
using Random: seed!
using Roots
using Requires
using Setfield
using StaticArrays: @SMatrix, @SVector, SMatrix, StaticArray, StaticArrayStyle,
    StaticMatrix, StaticVector, SVector
using Statistics
using StatsBase
using Strided

import Base: +, -, *, \, /, ^, ~, ≈,
    adjoint, axes, broadcast, broadcastable, BroadcastStyle, conj, convert,
    copy, copyto!, eltype, fill!, getindex, getproperty, hash, hcat, hvcat, inv,
    iterate, keys, lastindex, length, literal_pow, materialize!, materialize,
    one, permutedims, print_array, promote, promote_rule, promote_rule,
    promote_type, propertynames, real, setindex!, show, show_datatype,
    show_vector, similar, size, sqrt, string, summary, transpose, zero
import Base.Broadcast: instantiate, preprocess
import LinearAlgebra: dot, isnan, ldiv!, logdet, mul!, pinv,
    StructuredMatrixStyle, structured_broadcast_alloc, tr
import PyPlot: loglog, plot, semilogx, semilogy



export
    @namedtuple, @repeated, @unpack, animate, argmaxf_lnP, azeqproj, BandPassOp,
    cache, CachedLenseFlow, camb, cg, class, cov_to_Cℓ, Cℓ_2D, Cℓ_to_Cov, DataSet,
    DerivBasis, Diagonal, DiagOp, dot, EBFourier, EBMap, FFTgrid, Field, FieldArray,
    FieldMatrix, FieldOrOpArray, FieldOrOpMatrix, FieldOrOpRowVector,
    FieldOrOpVector, FieldRowVector, FieldTuple, FieldVector, FieldVector, Flat,
    FlatEB, FlatEBFourier, FlatEBMap, FlatFourier, FlatIEBCov, FlatIEBFourier,
    FlatIEBMap, flatinfo, FlatIQUFourier, FlatIQUMap, FlatMap, FlatQU,
    FlatQUFourier, FlatQUMap, FlatS0, FlatS02, FlatS2, FlatS2Fourier, FlatS2Map,
    Fourier, fourier∂, FuncOp, FΦTuple, get_Cℓ, get_Cℓ, get_Dℓ, get_αℓⁿCℓ,
    get_ρℓ, get_ℓ⁴Cℓ, gradhess, gradient, GradientCache, HealpixCap,
    HealpixS0Cap, HealpixS2Cap, HighPass, IdentityOp, IEBFourier, IEBMap,
    InterpolatedCℓs, IQUFourier, IQUMap, IsotropicHarmonicCov, LazyBinaryOp,
    LenseBasis, LenseFlow, LenseOp, LinDiagOp, LinOp, lnP,
    load_healpix_sim_dataset, load_sim_dataset, LowPass, make_mask, Map,
    MAP_joint, MAP_marg, map∂, MidPass, mix, nan2zero, noiseCℓs, NoLensing,
    OuterProdOp, pack, ParamDependentOp, pixwin, plot, PowerLens,
    quadratic_estimate, QUFourier, QUMap, resimulate, RK4Solver, S0, S02, S2,
    sample_joint, shortname, simulate, symplectic_integrate, Taylens, toCℓ,
    toDℓ, tuple_adjoint, ud_grade, unmix, Ð, Ł, δf̃ϕ_δfϕ, δfϕ_δf̃ϕ, δlnP_δfϕₜ,
    ℓ², ℓ⁴, ∇, ∇², ∇¹, ∇ᵢ, ∇⁰, ∇ⁱ, ∇₀, ∇₁, ⋅, ⨳   
    
# generic stuff
include("util.jl")
include("util_fft.jl")
include("numerical_algorithms.jl")
include("generic.jl")
include("cls.jl")
include("field_tuples.jl")
include("field_vectors.jl")
include("specialops.jl")

# lensing
include("lensing.jl")
include("lenseflow.jl")
include("powerlens.jl")

# flat-sky maps
include("flat_fftgrid.jl")
include("flat_s0.jl")
include("flat_s2.jl")
include("flat_s0s2.jl")
include("flat_generic.jl")
include("masking.jl")
include("taylens.jl")

# plotting
include("plotting.jl")

# sampling and maximizing the posteriors
include("dataset.jl")
include("posterior.jl")
include("sampling.jl")

# other estimates
include("quadratic_estimate.jl")

# curved-sky (not yet upgraded to new system)
# include("healpix.jl")

include("autodiff.jl")

end
