module CMBLensing

using Base.Iterators: repeated
using Base.Threads
using Interpolations
using MacroTools
using NamedTuples
using ODE
using Parameters
using ProgressMeter
using PyCall
using Requires
using StaticArrays
using StatsBase
include("RFFTVectors.jl"); using .RFFTVectors
include("masking.jl"); using .Masking


import Base: +, -, *, \, /, ^, ~, .*, ./, .^,
    Ac_mul_B, Ac_ldiv_B, broadcast, convert, done, eltype, getindex,
    inv, length, literal_pow, next, promote_rule, size, sqrt, start, transpose, ctranspose, one, zero, sqrtm
import Base.LinAlg: dot, norm, isnan


function __init__()
    global classy = pyimport("classy")
    # global hp = pyimport("healpy")
end


export
    Field, LinOp, LinDiagOp, FullDiagOp, Ð, Ł, simulate, Cℓ_to_cov,
    S0, S2, S02, Map, Fourier,
    ∂x, ∂y, ∇, ∇²,
    Cℓ_2D, ⨳, @⨳, shortname, Squash, IdentityOp, pixstd, ud_grade

include("util.jl")
include("generic.jl")
include("specialops.jl")
include("algebra.jl")
include("field_tuples.jl")
include("lensing.jl")
include("flat.jl")
include("taylens.jl")
include("vec_conv.jl")
include("healpix.jl")
@require PyPlot include("plotting.jl")
include("cls.jl")
include("likelihood.jl")
include("wiener.jl")
include("minimize.jl")

end
