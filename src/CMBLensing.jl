module CMBLensing

using Base.Broadcast: AbstractArrayStyle, ArrayStyle, Broadcasted, broadcasted, DefaultArrayStyle, flatten, preprocess_args, Style
using Base.Iterators: repeated
using Base.Threads
using Base: @propagate_inbounds, show_vector, show_default
using CatViews
using Combinatorics
using DataStructures
using Distributed
using FFTW
using Images: feature_transform, imfilter
using Images.Kernel
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




import Base: +, -, *, \, /, ^, ~,
    adjoint, broadcast, broadcastable, BroadcastStyle, convert, copy, copyto!,
    eltype, getindex, getproperty, inv, iterate, keys, length, literal_pow,
    materialize!, materialize, one, print_array, promote,
    promote_rule, promote_rule, promote_type, propertynames, real, setindex!,
    show, showarg,
    similar, size, sqrt, sqrt, summary, transpose, zero
import Base.Broadcast: instantiate, preprocess
import LinearAlgebra: dot, isnan, ldiv!, logdet, mul!
import PyPlot: loglog, plot, semilogx, semilogy



export
    Field, LinOp, LinDiagOp, FullDiagOp, Ð, Ł, simulate, Cℓ_to_cov, cov_to_Cℓ,
    S0, S2, S02, Map, Fourier,
    ∇⁰, ∇¹, ∇₀, ∇₁, ∇, ∇ⁱ, ∇ᵢ, ∇²,
    Cℓ_2D, ⨳, shortname, Squash, IdentityOp, ud_grade,
    get_Cℓ, get_Dℓ, get_αℓⁿCℓ, get_ℓ⁴Cℓ, get_ρℓ, 
    BandPassOp, FuncOp, lensing_wiener_filter, animate, symplectic_integrate,
    MAP_joint, MAP_marg, sample_joint, load_sim_dataset, norm², pixwin,
    HealpixS0Cap, HealpixS2Cap, HealpixCap, GradientCache, azeqproj, HighPass, LowPass, MidPass,
    plot, @unpack, OuterProdOp, resimulate,
    ℓ², ℓ⁴, toCℓ, toDℓ, InterpolatedCℓs, ParamDependentOp,
    IsotropicHarmonicCov, load_healpix_sim_dataset, dot, ⋅, cache, fourier∂, map∂, Diagonal

include("util.jl")
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
# include("minimize.jl")
# include("masking.jl")
# include("quadratic_estimate.jl")
# include("properties.jl")
# include("weave_pyplot.jl")

end
