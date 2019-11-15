using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--tests"
        default = false
        arg_type = Bool
    "--benchmarks"
        default = true
        arg_type = Bool
    "--storage"
        help = "an option without argument, i.e. a flag"
        range_tester = s -> s in ("Array","CuArray")
        default = "Array"
    "--benchmark_accuracy"
        default = 0.01
        arg_type = Float64
    "--N"
        default = 256
        arg_type = Int
    "--T"
        default = Float32
        range_tester = s -> s in (Float32,Float64)
        eval_arg = true
end
args = parse_args(ARGS, s)

args["storage"] = "CuArray"

if args["storage"]=="CuArray"
    using CuArrays
end
storage = eval(Symbol(args["storage"]))
macro belapsed(ex)
    if args["storage"] == "CuArray" 
        esc(:(BenchmarkTools.@belapsed CuArrays.@sync $ex))
    else
        esc(:(BenchmarkTools.@belapsed $ex))
    end
end
N = args["N"]
T = args["T"]

##

using CMBLensing
using CMBLensing: adapt

##

using Base.Broadcast: broadcastable
import BenchmarkTools
using Crayons
using DataStructures
using FileIO
using LibGit2
using PrettyTables
using Printf
using Test


##

# benchmark to 1% and check to 5%, so only a "5σ event" should trigger a failure
BenchmarkTools.DEFAULT_PARAMETERS.time_tolerance = args["benchmark_accuracy"]
rtol = 0.05
≲(a,b; rtol=rtol) = (a < (1+rtol)*b)

##

if args["tests"]
try 
    
##

@testset "Performance Tests" begin

# spin 0 
f = adapt(storage,FlatMap(rand(T,N,N)))
g = adapt(storage,FlatMap(rand(T,N,N)))
Ðf = Ð(f)
@test @belapsed(@.      $f + $f) ≲ @belapsed(@.         $f.Ix + $f.Ix)
@test @belapsed(@. $g = $f + $f) ≲ @belapsed(@. $g.Ix = $f.Ix + $f.Ix)
@test @belapsed(∇[1]*$Ðf)        ≲ @belapsed($(broadcastable(typeof(Ðf), ∇[1].diag)) .* $Ðf.Il)

# spin 2
ft = adapt(storage,FlatQUMap(rand(T,N,N),rand(T,N,N)))
gt = adapt(storage,FlatQUMap(rand(T,N,N),rand(T,N,N)))
@test @belapsed(@.       $ft + $ft) ≲ @belapsed(@. (         $ft.Qx + $ft.Qx,          $ft.Ux + $ft.Ux))
@test @belapsed(@. $gt = $ft + $ft) ≲ @belapsed(@. ($gt.Qx = $ft.Qx + $ft.Qx, $gt.Ux = $ft.Ux + $ft.Ux))

end

##

catch 
end
end

##

if args["benchmarks"]

Cℓ = camb().unlensed_total

# spin 0
f = adapt(storage,simulate(Cℓ_to_Cov(Flat(Nside=N), T, S0, Cℓ.TT)))
ϕ = adapt(storage,simulate(Cℓ_to_Cov(Flat(Nside=N), T, S0, Cℓ.ϕϕ)))
Lϕ = cache(LenseFlow(ϕ),f)

tL0  = @belapsed($Lϕ *$f)
tLt0 = @belapsed($Lϕ'*$f)
tgL0 = @belapsed($(δf̃ϕ_δfϕ(Lϕ,f,f)')*$(FΦTuple(f,ϕ)))

# spin 2
f = adapt(storage,simulate(Cℓ_to_Cov(Flat(Nside=N), T, S2, Cℓ.EE, Cℓ.BB)))
ϕ = adapt(storage,simulate(Cℓ_to_Cov(Flat(Nside=N), T, S0, Cℓ.ϕϕ)))
Lϕ = cache(LenseFlow(ϕ),f)

tL2  = @belapsed($Lϕ *$f)
tLt2 = @belapsed($Lϕ'*$f)
tgL2 = @belapsed($(δf̃ϕ_δfϕ(Lϕ,f,f)')*$(FΦTuple(f,ϕ)))

##


meta = OrderedDict(
    "COMMIT" => string(LibGit2.head_oid(GitRepo(joinpath(dirname(@__FILE__),".."))))[1:7],
    "JULIA_NUM_THREADS" => Base.Threads.nthreads(),
    "FFTW_NUM_THREADS" => CMBLensing.FFTW_NUM_THREADS[],
    (k=>args[k] for k in ["storage","N","T"])...
)

timing = [
    "Spin-0 LenseFlow"           tL0   0.013;
    "Spin-0 Adjoint LenseFlow"   tLt0  0.013;
    "Spin-0 Gradient LenseFlow"  tgL0  0.080;
    "Spin-2 LenseFlow"           tL2   0.030;
    "Spin-2 Adjoint LenseFlow"   tLt2  0.030;
    "Spin-2 Gradient LenseFlow"  tgL2  0.140;
]



# print benchmarks

pretty_table(Dict(meta), crop=:none)

pretty_table(
    timing,
    ["Operation","Time","Reference (Cori-Haswell)"],
    formatter = Dict(
        1 => (v,_) -> v,
        2 => (v,_) -> @sprintf("%.1f ms",1000v),
        3 => (v,_) -> @sprintf("%.0f ms",1000v)
    ),
    highlighters = (
        Highlighter((data, i, j) -> j==2 && data[i,j] > (1+rtol) * data[i,j+1], foreground=:red),
        Highlighter((data, i, j) -> j==2 && data[i,j] < (1-rtol) * data[i,j+1], foreground=:green)
    ),
    crop=:none
)

# save benchmarks
filename = "benchmarks/"*join(["$(k)_$(v)" for (k,v) in meta], "__")*".jld2"
!ispath("benchmarks") && mkdir("benchmarks")
save(filename,"meta",meta,"timing",timing)


end
