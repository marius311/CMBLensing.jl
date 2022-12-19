
module CMBLensing

using Adapt
using Base.Broadcast: AbstractArrayStyle, ArrayStyle, Broadcasted,
    DefaultArrayStyle, preprocess_args, Style, result_style, Unknown
using Base.Iterators: flatten, product, repeated, cycle, countfrom, peel, partition
using Base: @kwdef, @propagate_inbounds, Bottom, OneTo, showarg, show_datatype,
    show_default, show_vector, typed_vcat, typename
using ChainRules
using Combinatorics
using CompositeStructs
using CoordinateTransformations
using DataStructures
using DelimitedFiles
using Distributed: pmap, nworkers, myid, workers, addprocs, @everywhere, remotecall_wait, 
    @spawnat, pgenerate, procs, @fetchfrom, default_worker_pool, RemoteChannel
using Distributions
using Distributions: PDiagMat
using EllipsisNotation
using FileIO
using FFTW
using ForwardDiff
using Healpix
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
using PDMats
using Pkg
using Preferences
using Printf
using ProgressMeter
using QuadGK
using Random
using Random: seed!, AbstractRNG
using Rotations
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
using Tullio
using UnPack
using Zygote
using Zygote: unbroadcast, Numeric, @adjoint, @nograd
using Zygote.ChainRules: @thunk, NoTangent


import Adapt: adapt_structure
import Base: +, -, *, \, /, ^, ~, ≈, <, <=, |, &, ==, !,
    abs, adjoint, all, any, axes, broadcast, broadcastable, BroadcastStyle, conj, copy, convert,
    copy, copyto!, eltype, eps, exp, fill!, getindex, getproperty, hash, haskey, hcat, hvcat, inv, isfinite,
    iterate, keys, lastindex, length, literal_pow, log, map, mapreduce, materialize!,
    materialize, merge, one, permutedims, print_array, promote, promote_rule,
    promote_rule, promote_type, propertynames, real, setindex!, setproperty!, show,
    show_datatype, show_vector, similar, size, sqrt, string, sum, summary,
    transpose, zero
import Base.Broadcast: materialize, preprocess, broadcasted
import Zygote.ChainRules: rrule
import LinearAlgebra: checksquare, diag, dot, isnan, ldiv!, logdet, mul!, norm,
    pinv, StructuredMatrixStyle, structured_broadcast_alloc, tr
import Measurements: ±
import Statistics: std
import ChainRules: ProjectTo
import Random: randn!


export
    @⌛, @show⌛, @ismain, @namedtuple, @repeated, @unpack, @cpu!, @gpu!, @cu!, @fwdmodel, 
    animate, argmaxf_lnP, argmaxf_logpdf, AzFourier, BandPassOp, BaseDataSet, batch, batch_index, batch_length, 
    batch_map, batch_pmap, BlockDiagEquiRect, beamCℓs, cache, CachedLenseFlow, camb, cov_to_Cℓ, cpu, Cℓ_2D, 
    Cℓ_to_Cov, DataSet, DerivBasis, diag, Diagonal, DiagOp, dot, EBFourier, EBMap, expnorm, 
    Field, FieldArray, fieldinfo, FieldMatrix, FieldOrOpArray, FieldOrOpMatrix, FieldOrOpRowVector,
    FieldOrOpVector, FieldRowVector, FieldTuple, FieldVector, FieldVector,
    firsthalf, BlockDiagIEB, Fourier, FuncOp, get_max_lensing_step,
    get_Cℓ, get_Cℓ, get_Dℓ, get_αℓⁿCℓ, get_ρℓ, get_ℓ⁴Cℓ, gradhess, gradient, HighPass,
    IEBFourier, IEBMap, Cℓs, IQUAzFourier, IQUFourier, IQUMap, kde,
    lasthalf, LazyBinaryOp, LenseBasis, LenseFlow, FieldOp, lnP, logpdf, load_camb_Cℓs,
    load_chains, load_nolensing_sim, load_sim, LowPass, make_mask, Map, MAP_joint, MAP_marg,
    mean_std_and_errors, MidPass, mix, Mixed, nan2zero, noiseCℓs,
    ParamDependentOp, pixwin, PowerLens, precompute!!, ProjLambert, ProjEquiRect, ProjHealpix, project,
    QUAzFourier, QUFourier, QUMap, resimulate!, resimulate, RK4Solver, sample_f, sample_joint, shiftℓ, 
    simulate, SymmetricFuncOp, symplectic_integrate, Taylens, toCℓ, toDℓ,
    ud_grade, unbatch, unmix, Ð, Ł,  
    ℓ², ℓ⁴, ∇, ∇², ∇ᵢ, ∇ⁱ

# bunch of sampling-related exports
export gibbs_initialize_f!, gibbs_initialize_ϕ!, gibbs_initialize_θ!, 
    gibbs_sample_f!, gibbs_sample_ϕ!, gibbs_sample_slice_θ!, 
    gibbs_mix!, gibbs_unmix!, gibbs_postprocess!, 
    once_every, start_after_burnin, mass_matrix_ϕ, hmc_step


# util
include("util.jl")
include("util_fft.jl")
include("util_parallel.jl")
include("numerical_algorithms.jl")

# generic field / operator structure
include("generic.jl")
include("cls.jl")
include("field_tuples.jl")
include("field_vectors.jl")
include("base_fields.jl")
include("specialops.jl")
include("flowops.jl")
include("batching.jl")

# lensing operators
include("lenseflow.jl")
include("powerlens.jl")

# field types
include("proj_cartesian.jl")
include("proj_lambert.jl")
include("proj_equirect.jl")
include("proj_healpix.jl")

# other field-specific stuff
include("masking.jl")
include("taylens.jl")
include("bilinearlens.jl")

# plotting
function animate end
@init @require PyPlot="d330b81b-6aea-500a-939a-2ce795aea3ee" include("pyplot.jl")
include("plots.jl")

# PPL
include("distributions.jl")
include("simpleppl.jl")

# sampling and maximizing the posteriors
include("dataset.jl")
include("maximization.jl")
include("sampling.jl")
include("chains.jl")

# deprecated stuff
include("deprecated.jl")

# other estimates
include("quadratic_estimate.jl")

# AD
include("autodiff.jl")

# gpu
is_gpu_backed(x) = false
@init @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("gpu.jl")

# muse
@init @require MuseInference="43b88160-90c7-4f71-933b-9d65205cd921" include("muse.jl")

# misc init
# see https://github.com/timholy/ProgressMeter.jl/issues/71 and links therein
@init if ProgressMeter.@isdefined ijulia_behavior
    ProgressMeter.ijulia_behavior(:clear)
end

end

