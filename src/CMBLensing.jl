module CMBLensing

using Base.Iterators: repeated
using Base.Threads
using DataStructures
using FFTW
using Images: feature_transform, imfilter
using Images.Kernel
using InteractiveUtils
using Interpolations
using Lazy: @switch, @init
using LinearAlgebra
using MacroTools: @capture, postwalk, isexpr
using Markdown
using Optim: optimize
using Parameters
using ProgressMeter
using PyCall
using PyPlot
using Random
using Random: seed!
using StaticArrays: StaticArray, StaticVector, StaticMatrix, SVector, SMatrix, @SVector, @SMatrix
using Statistics
using StatsBase
include("RFFTVectors.jl")
using .RFFTVectors

@init @eval @pyimport healpy as hp
@init @eval @pyimport pylab

import Base: +, -, *, \, /, ^, ~, .*, ./, .^, broadcast,
    convert, copy, done, eltype, full, getindex, getproperty, propertynames, inv, length, literal_pow, next, 
    promote_rule, similar, size, sqrt, start, transpose, one, zero,
    sqrt, adjoint, promote, promote_rule
import LinearAlgebra: dot, isnan, logdet



export
    Field, LinOp, LinDiagOp, FullDiagOp, Ð, Ł, simulate, Cℓ_to_cov,
    S0, S2, S02, Map, Fourier,
    ∇⁰, ∇¹, ∇₀, ∇₁, ∇, ∇ⁱ, ∇ᵢ, Δ,
    Cℓ_2D, ⨳, @⨳, shortname, Squash, IdentityOp, ud_grade,
    get_Cℓ, get_Dℓ, get_αℓⁿCℓ, get_ℓ⁴Cℓ, get_ρℓ, 
    BandPassOp, FuncOp, lensing_wiener_filter, animate, symplectic_integrate,
    max_lnP_joint, load_sim_dataset, norm², pixwin

include("util.jl")
include("generic.jl")
include("specialops.jl")
include("algebra.jl")
include("smashtimes.jl")
include("field_tuples.jl")
include("lensing.jl")
include("flat.jl")
include("healpix.jl")
include("taylens.jl")
include("vec_conv.jl")
include("plotting.jl")
include("cls.jl")
include("likelihood.jl")
include("sampling.jl")
include("minimize.jl")
include("masking.jl")
include("quadratic_estimate.jl")

end
