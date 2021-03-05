
module CMBLensing

using Adapt
using Base.Broadcast: AbstractArrayStyle, ArrayStyle, Broadcasted,
    DefaultArrayStyle, preprocess_args, Style, result_style, Unknown
using Base.Iterators: flatten, product, repeated, cycle, countfrom, peel
using Base.Threads
using Base: @kwdef, @propagate_inbounds, Bottom, OneTo, showarg, show_datatype,
    show_default, show_vector, typed_vcat, typename
using Combinatorics
using DataStructures
using DelimitedFiles
using Distributed: pmap, nworkers, myid, workers, addprocs, @everywhere, remotecall_wait, 
    @spawnat, pgenerate, procs, @fetchfrom, default_worker_pool
using FileIO
using FFTW
using InteractiveUtils
using IterTools: flagfirst
using JLD2
using JLD2: jldopen, JLDWriteSession
using KahanSummation
using Loess
using LinearAlgebra
using LinearAlgebra: diagzero, matprod, promote_op
using MacroTools: @capture, combinedef, isdef, isexpr, postwalk, prewalk, rmlines, splitdef
using Match
using Markdown
using Measurements
using Memoization
using NamedTupleTools: select, delete
using OptimKit
using Pkg
using Printf
using ProgressMeter
using QuadGK
using Random
using Random: seed!, AbstractRNG
using Roots
using Requires
using Serialization
using Setfield
using SparseArrays
using StaticArrays: @SMatrix, @SVector, SMatrix, StaticArray, StaticArrayStyle,
    StaticMatrix, StaticVector, SVector, SArray, SizedArray
using Statistics
using StatsBase
using TimerOutputs: @timeit, get_defaulttimer, reset_timer!
using UnPack
using Zygote
using Zygote: unbroadcast, Numeric, @adjoint, @nograd


import Adapt: adapt_structure
import Base: +, -, *, \, /, ^, ~, ≈, <, <=, |, &, ==, !,
    abs, adjoint, all, any, axes, broadcast, broadcastable, BroadcastStyle, conj, copy, convert,
    copy, copyto!, eltype, eps, fill!, getindex, getproperty, hash, hcat, hvcat, inv, isfinite,
    iterate, keys, lastindex, length, literal_pow, mapreduce, materialize!,
    materialize, one, permutedims, print_array, promote, promote_rule,
    promote_rule, promote_type, propertynames, real, setindex!, setproperty!, show,
    show_datatype, show_vector, similar, size, sqrt, string, sum, summary,
    transpose, zero
import Base.Broadcast: materialize, preprocess, broadcasted
import LinearAlgebra: checksquare, diag, dot, isnan, ldiv!, logdet, mul!, norm,
    pinv, StructuredMatrixStyle, structured_broadcast_alloc, tr
import Measurements: ±
import Statistics: std


export
    @⌛, @show⌛, @ismain, @namedtuple, @repeated, @unpack, animate,
    argmaxf_lnP, BandPassOp, BaseDataSet, batch, batch_index, batch_length, beamCℓs, cache,
    CachedLenseFlow, camb, cov_to_Cℓ, cpu, Cℓ_2D, Cℓ_to_Cov, DataSet, DerivBasis,
    diag, Diagonal, DiagOp, dot, EBFourier, EBMap, expnorm, Field, FieldArray, fieldinfo,
    FieldMatrix, FieldOrOpArray, FieldOrOpMatrix, FieldOrOpRowVector,
    FieldOrOpVector, FieldRowVector, FieldTuple, FieldVector, FieldVector,
    firsthalf, fixed_white_noise, FlatEB, FlatEBFourier, FlatEBMap, FlatField, 
    FlatFourier, FlatIEBCov, FlatIEBFourier, FlatIEBMap,
    FlatIQUFourier, FlatIQUMap, FlatMap, FlatQU, FlatQUFourier, FlatQUMap, FlatS0,
    FlatS02, FlatS2, FlatS2Fourier, FlatS2Map, Fourier, FuncOp, get_Cℓ,
    get_Cℓ, get_Dℓ, get_αℓⁿCℓ, get_ρℓ, get_ℓ⁴Cℓ, gradhess, gradient, HighPass,
    IEBFourier, IEBMap, InterpolatedCℓs, IQUFourier, IQUMap, kde,
    lasthalf, LazyBinaryOp, LenseBasis, LenseFlow, FieldOp, lnP, load_camb_Cℓs,
    load_chains, load_sim, LowPass, make_mask, Map, MAP_joint, MAP_marg,
    mean_std_and_errors, MidPass, mix, nan2zero, new_dataset, noiseCℓs,
    ParamDependentOp, pixwin, PowerLens, ProjLambert, QUFourier, QUMap, resimulate!,
    resimulate, RK4Solver, sample_joint, shiftℓ, 
    simulate, SymmetricFuncOp, symplectic_integrate, Taylens, toCℓ, toDℓ,
    ud_grade, unbatch, unmix, white_noise, Ð, Ł,  
    ℓ², ℓ⁴, ∇, ∇², ∇ᵢ, ∇ⁱ

# bunch of sampling-related exports
export gibbs_initialize_f!, gibbs_initialize_ϕ!, gibbs_initialize_θ!, 
    gibbs_sample_f!, gibbs_sample_ϕ!, gibbs_sample_slice_θ!, 
    gibbs_mix!, gibbs_unmix!, gibbs_postprocess!, 
    once_every, start_after_burnin, mass_matrix_ϕ, hmc_step


# generic stuff
include("util.jl")
include("util_fft.jl")
include("numerical_algorithms.jl")
include("generic.jl")
include("cls.jl")
include("field_tuples.jl")
include("field_vectors.jl")
include("specialops.jl")
include("flowops.jl")
include("batched_reals.jl")
include("base_fields.jl")

# # lensing
include("lenseflow.jl")
include("powerlens.jl")

# # flat-sky maps
include("flat_proj.jl")
include("flat_fields.jl")
include("masking.jl")
include("taylens.jl")
include("bilinearlens.jl")

# plotting
function animate end
@init @require PyPlot="d330b81b-6aea-500a-939a-2ce795aea3ee" include("plotting.jl")

# # sampling and maximizing the posteriors
include("dataset.jl")
include("posterior.jl")
include("maximization.jl")
include("sampling.jl")
include("chains.jl")

# other estimates
include("quadratic_estimate.jl")

# curved-sky
include("healpix.jl")

include("autodiff.jl")

# gpu
is_gpu_backed(x) = false
@init @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("gpu.jl")

# misc init
# see https://github.com/timholy/ProgressMeter.jl/issues/71 and links therein
@init if ProgressMeter.@isdefined ijulia_behavior
    ProgressMeter.ijulia_behavior(:clear)
end

end

