push!(LOAD_PATH, pwd()*"/src")

using CMBLensing
using Test

macro test_noerr(ex) :(@test ($(esc(ex)); true)) end
