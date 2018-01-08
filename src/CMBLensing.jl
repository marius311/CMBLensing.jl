__precompile__()

module CMBLensing

using Base.Iterators: repeated
using Base.Threads
using Images: feature_transform, imfilter, Kernel
using Interpolations
using MacroTools: @capture, postwalk, isexpr
using ODE
using Parameters
using ProgressMeter
using PyCall
using PyPlot
using StaticArrays: StaticArray, SMatrix, @SMatrix, SVector, @SVector
using StatsBase
include("RFFTVectors.jl"); using .RFFTVectors



import Base: +, -, *, \, /, ^, ~, .*, ./, .^, Ac_mul_B, Ac_ldiv_B, broadcast,
    convert,copy, done, eltype, getindex, inv, length, literal_pow, next,
    promote_rule, similar, size, sqrt, start, transpose, ctranspose, one, zero,
    sqrtm
import Base.LinAlg: dot, norm, isnan



export
    Field, LinOp, LinDiagOp, FullDiagOp, Ð, Ł, simulate, Cℓ_to_cov,
    S0, S2, S02, Map, Fourier,
    ∂x, ∂y, ∇, ∇²,
    Cℓ_2D, ⨳, @⨳, shortname, Squash, IdentityOp, pixstd, ud_grade,
    get_Cℓ, get_Dℓ, get_αℓⁿCℓ, BandPassOp, FuncOp

include("util.jl")
include("generic.jl")
include("specialops.jl")
include("algebra.jl")
include("field_tuples.jl")
include("lensing.jl")
include("flat.jl")
include("taylens.jl")
include("vec_conv.jl")
include("plotting.jl")
include("cls.jl")
include("likelihood.jl")
include("minimize.jl")
include("masking.jl")
include("quadratic_estimate.jl")

end
