push!(LOAD_PATH, pwd()*"/src")

using CMBLensing
using Base.Test

macro test_noerror(ex) :(@test ($(esc(ex)); true)) end

include("algebra.jl")
include("conversions.jl")
include("dotproducts.jl")
