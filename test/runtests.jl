push!(LOAD_PATH, pwd()*"/../src")

using CMBFields
using Base.Test

macro test_noerror(ex) :(@test ($(esc(ex)); true)) end

include("algebra.jl")
