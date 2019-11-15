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

import BenchmarkTools
using Base.Broadcast: broadcastable
using Crayons
using DelimitedFiles
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

Cℓ = camb()
timing = []

for (s,use) in [(0,:I),(2,:P)]
    
    @unpack f,f̃,ϕ,ds,ds₀ = load_sim_dataset(θpix=3, Nside=N, T=Float32, use=use, storage=storage, Cℓ=Cℓ);
    f°,ϕ° = mix(f,ϕ,ds₀)
    Lϕ = cache(LenseFlow(ϕ),f)
    
    append!(timing,[
        "Spin-$s Cache L" => @belapsed cache(LenseFlow($ϕ), $f);
        "Spin-$s L"       => @belapsed $Lϕ  * $f;
        "Spin-$s L†"      => @belapsed $Lϕ' * $f;
        "Spin-$s (∇L)†"   => @belapsed $(δf̃ϕ_δfϕ(Lϕ,f,f)') * $(FΦTuple(f,ϕ));
        "Spin-$s lnP"     => @belapsed       lnP(:mix, $f°, $ϕ°, $ds₀, $Lϕ);
        "Spin-$s ∇lnP"    => @belapsed δlnP_δfϕₜ(:mix, $f°, $ϕ°, $ds₀, $Lϕ);
    ])

end

meta = [
    "COMMIT" => string(LibGit2.head_oid(GitRepo(joinpath(dirname(@__FILE__),".."))))[1:7],
    "JULIA_NUM_THREADS" => Base.Threads.nthreads(),
    "FFTW_NUM_THREADS" => CMBLensing.FFTW_NUM_THREADS[],
    (k=>args[k] for k in ["storage","N","T"])...
]

reference_timing = Dict(
    "Spin-0 Cache L" => 25,
    "Spin-0 L"       => 13,
    "Spin-0 L†"      => 13,
    "Spin-0 (∇L)†"   => 85,
    "Spin-0 lnP"     => 65,
    "Spin-0 ∇lnP"    => 240,
    "Spin-2 Cache L" => 25,
    "Spin-2 L"       => 30,
    "Spin-2 L†"      => 30,
    "Spin-2 (∇L)†"   => 140,
    "Spin-2 lnP"     => 110,
    "Spin-2 ∇lnP"    => 380
)


# print benchmarks

pretty_table(Dict(meta), crop=:none)

pretty_table(
    vcat(([k v reference_timing[k]*1e-3] for (k,v) in timing)...),
    ["Operation","Time","Reference"],
    formatter = Dict(
        1 => (v,_) -> v,
        2 => (v,_) -> @sprintf("%.1f ms",1000v),
        3 => (v,_) -> @sprintf("%.0f ms",1000v)
    ),
    highlighters = (
        Highlighter((data, i, j) -> j==2 && data[i,j] > (1+rtol) * data[i,j+1], foreground=:red),
        Highlighter((data, i, j) -> j==2 && data[i,j] < (1-rtol) * data[i,j+1], foreground=:green)
    ),
    crop=:none,
    alignment=[:l,:r,:r],
    hlines=[6]
)

# save benchmarks
filename = "benchmarks/"*join(["$(k)_$(v)" for (k,v) in meta], "__")*".txt"
!ispath("benchmarks") && mkdir("benchmarks")
open(filename,"w") do f
    writedlm.(Ref(f),(meta,timing))
end

end
