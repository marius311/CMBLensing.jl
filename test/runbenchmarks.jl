using CMBLensing

##

using Base.Broadcast: broadcastable
using BenchmarkTools
using BenchmarkTools: @belapsed, @btime
using Crayons
using FileIO
using LibGit2
using PrettyTables
using Printf
using Test


# benchmark to 1% and check to 5%, so only a "5σ event" should trigger a failure
BenchmarkTools.DEFAULT_PARAMETERS.time_tolerance = 0.01
rtol = 0.05

≲(a,b; rtol=rtol) = (a < (1+rtol)*b)

N = 256

##

@testset "Benchmarks" begin

# broadcasting with Fields should be exactly as fast working with the arrays
# directly, no overhead allowed

@testset "Algebra" begin

# spin 0 
f = FlatMap(rand(N,N))
g = FlatMap(rand(N,N))
Ðf = Ð(f)
@test @belapsed(@.      $f + $f) ≲ @belapsed(@.         $f.Ix + $f.Ix)
@test @belapsed(@. $g = $f + $f) ≲ @belapsed(@. $g.Ix = $f.Ix + $f.Ix)
@test @belapsed(∇[1]*$Ðf)        ≲ @belapsed($(broadcastable(typeof(Ðf), ∇[1].diag)) .* $Ðf.Il)

# spin 2
ft = FlatQUMap(rand(N,N),rand(N,N))
gt = FlatQUMap(rand(N,N),rand(N,N))
@test @belapsed(@.       $ft + $ft) ≲ @belapsed(@. (         $ft.Qx + $ft.Qx,          $ft.Ux + $ft.Ux))
@test @belapsed(@. $gt = $ft + $ft) ≲ @belapsed(@. ($gt.Qx = $ft.Qx + $ft.Qx, $gt.Ux = $ft.Ux + $ft.Ux))

end

##

Cℓ = camb().unlensed_total

# spin 0
f = simulate(Cℓ_to_Cov(Flat(Nside=N), Float32, S0, Cℓ.TT))
ϕ = simulate(Cℓ_to_Cov(Flat(Nside=N), Float32, S0, Cℓ.ϕϕ))
Lϕ = cache(LenseFlow(ϕ),f)

tL0  = @belapsed($Lϕ *$f))
tLt0 = @belapsed($Lϕ'*$f))
tgL0 = @belapsed($(δf̃ϕ_δfϕ(Lϕ,f,f)')*$(FΦTuple(f,ϕ))))

# spin 2
f = simulate(Cℓ_to_Cov(Flat(Nside=N), Float32, S2, Cℓ.EE, Cℓ.BB))
ϕ = simulate(Cℓ_to_Cov(Flat(Nside=N), Float32, S0, Cℓ.ϕϕ))
Lϕ = cache(LenseFlow(ϕ),f)

tL2  = @belapsed($Lϕ *$f))
tLt2 = @belapsed($Lϕ'*$f))
tgL2 = @belapsed($(δf̃ϕ_δfϕ(Lϕ,f,f)')*$(FΦTuple(f,ϕ))))

##


meta = OrderedDict(
    "COMMIT" => string(LibGit2.head_oid(GitRepo(joinpath(dirname(@__FILE__),".."))))[1:7],
    "JULIA_NUM_THREADS" => Base.Threads.nthreads(),
    "FFTW_NUM_THREADS" => CMBLensing.FFTW_NUM_THREADS[],
)

timing = [
    "Spin-0 LenseFlow"           tL0   0.012;
    "Spin-0 Adjoint LenseFlow"   tLt0  0.013;
    "Spin-0 Gradient LenseFlow"  tgL0  0.080;
    "Spin-2 LenseFlow"           tL2   0.030;
    "Spin-2 Adjoint LenseFlow"   tLt2  0.030;
    "Spin-2 Gradient LenseFlow"  tgL2  0.140;
]

# save benchmarks
filename = "benchmarks/"*join(["$(k)_$(v)" for (k,v) in meta], "__")*".jld2"
!ispath("benchmarks") && mkdir("benchmarks")
save(filename,"meta",meta,"timing",timing)


# print benchmarks

pretty_table(Dict(meta))

pretty_table(
    timing,
    ["Operation","Time","Fiducial"],
    formatter = Dict(
        1 => (v,_) -> v,
        2 => (v,_) -> @sprintf("%.1f ms",1000v),
        3 => (v,_) -> @sprintf("%.0f ms",1000v)
    ),
    highlighters = (
        Highlighter((data, i, j) -> j==2 && data[i,j] > (1+rtol) * data[i,j+1], foreground=:red),
        Highlighter((data, i, j) -> j==2 && data[i,j] < (1-rtol) * data[i,j+1], foreground=:green)
    )
)
