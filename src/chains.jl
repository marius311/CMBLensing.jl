import Base: getindex, lastindex


@doc doc"""
    load_chains(filename; burnin=0, burnin_chunks=0, thin=1, join=false, unbatch=true)

Load a single chain or multiple parallel chains which were written to
a file by [`sample_joint`](@ref). 

Keyword arguments: 

* `burnin` — Remove this many samples from the start of each chain, or
  if negative, keep only this many samples at the end of each chain.
* `burnin_chunks` — Same as burnin, but in terms of chain "chunks"
  stored in the chain file, rather than in terms of samples.
* `thin` — If `thin` is an integer, thin the chain by this factor. If
  `thin == :hasmaps`, return only samples which have maps saved. If
  thin is a `Function`, filter the chain by this function (e.g.
  `thin=haskey(:g)` on Julia 1.5+)
* `unbatch` — If true, [unbatch](@ref) the chains if they are batched.
* `join` — If true, concatenate all the chains together.
* `skip_missing_chunks` — Skip missing chunks in the chain instead of
  terminating the chain there. 


The object returned by this function is a `Chain` or `Chains` object,
which simply wraps an `Array` of `Dicts` or an `Array` of `Array` of
`Dicts`, respectively (each sample is a `Dict`). The wrapper object
has some extra indexing properties for convenience: 

* It can be indexed as if it were a single multidimensional object,
  e.g. `chains[1,:,:accept]` would return the `:accept` key of all
  samples in the first chain.
* Leading colons can be dropped, i.e. `chains[:,:,:accept]` is the
  same as `chains[:accept]`. 
* If some samples are missing a particular key, `missing` is returned
  for those samples insted of an error.
* The recursion goes arbitrarily deep into the objects it finds. E.g.,
  since sampled parameters are stored in a `NamedTuple` like
  `(Aϕ=1.3,)` in the `θ` key of each sample `Dict`, you can do
  `chain[:θ,:Aϕ]` to get all `Aϕ` samples as a vector. 


"""
function load_chains(
    filename; 
    burnin = 0,
    thin = 1,
    join = false,
    unbatch = true,
    dropmaps = false,
    burnin_chunks = 0,
    progress = true
)
    chains = jldopen(filename) do io
        ks = keys(io)
        chunk_ks = sort([k for k in ks if startswith(k,"chunks_")], by=k->parse(Int,k[8:end]))
        chunk_ks = chunk_ks[burnin_chunks>=0 ? (burnin_chunks+1:end) : (end+burnin_chunks+1:end)]
        @showprogress (progress ? 1 : Inf) for (isfirst,k) in flagfirst(chunk_ks)
            if isfirst
                chains = read(io,k)
            else
                append!.(chains, read(io,k))
            end
            if dropmaps
                map!(chain -> filter!.(((_,v),)->!(v isa Field), chain), chains, chains)
            end
        end
        chains
    end
    if thin isa Int
        chains = [chain[burnin>=0 ? ((1+burnin):thin:end) : (end+(1+burnin):thin:end)] for chain in chains]
    elseif thin == :hasmaps
        chains = [[samp for samp in chain[(1+burnin):end] if :ϕ in keys(samp)] for chain in chains]
    elseif thin isa Function
        chains = [filter(thin,chain) for chain in chains]
    else
        error("`thin` should be an Int, :hasmaps, or a filter function")
    end
    chains = wrap_chains(chains)
    if unbatch
        chains = CMBLensing.unbatch(chains)
    end
    return join ? Chain(reduce(vcat, chains)) : chains
end


# multiple parallel chains
struct Chains{T} <: AbstractVector{T}
    chains :: Vector{T}
end
Chains(chains::Chains) = chains
getindex(c::Chains, k::Symbol, ks...) = _getindex(c.chains, :, :, k, ks...)
getindex(c::Chains, k,         ks...) = _getindex(c.chains, k, ks...)
size(c::Chains) = size(c.chains)
function Base.print_array(io::IO, cs::Chains; indent="  ")
    for c in cs
        print(io, "  ")
        Base.summary(io, c)
        println(io)
        Base.print_array(io, c)#, indent="    ")
    end
end
function lastindex(c::Chains, d)
    d==1 ? lastindex(c.chains) : 
    d==2 ? lastindex(c.chains[1]) : 
    error("`end` only valid in first or second dim of Chains")
end


# single chain
struct Chain{T} <: AbstractVector{T}
    chain :: Vector{T}
end
getindex(c::Chain, k::Symbol,              ks...) = _getindex(c.chain, :, k, ks...)
getindex(c::Chain, k,                      ks...) = _getindex(c.chain, k, ks...)
getindex(c::Chain, k::Union{Colon,AbstractRange}) = Chain(getindex(c.chain, k))
lastindex(c::Chain) = lastindex(c.chain)
lastindex(c::Chain, d) = d==1 ? lastindex(c.chain) : error("`end` only valid in first dim of Chain")
size(c::Chain) = size(c.chain)
function Base.print_array(io::IO, c::Chain; indent="  ")
    _,cols = displaysize(io)
    for k in keys(c[end])
        str = string("$(indent)$(k) => ", repr(c[k]; context=(:limit => true)))
        println(io, Base._truncate_at_width_or_chars(str, cols))
    end
end

# recurse
_getindex(x, k,                             k2, ks...) = _getindex(_getindex(x, k), k2, ks...)
_getindex(x, k::Union{Colon,AbstractRange}, k2, ks...) = broadcast(y -> _getindex(y, k2, ks...),  x[k])

# base cases
_getindex(x::Union{Dict,NamedTuple}, k::Symbol) = haskey(x,k) ? getindex(x, k) : missing
_getindex(x,                         k) = getindex(x, k)


wrap_chains(chains::Vector{<:Vector}) = Chains(Chain.(chains))
wrap_chains(chain::Vector) = Chain(chain)


# batching
@doc doc"""
    unbatch(chain::Chain)

Convert a chain of batch-length-`D` fields to `D` chains of unbatched fields. 
"""
function unbatch(chain::Chain)
    D = batch_length(chain[end][:logpdf])
    (D==1) && return [chain]
    Chains(map(1:D) do I
        Chain(map(chain) do samp
            Dict(map(collect(samp)) do (k,v)
                if v isa Union{BatchedReal,FlatField}
                    k => batch_index(v, I)
                elseif v isa NamedTuple{<:Any, <:NTuple{<:Any,<:Union{BatchedReal,BatchedVal}}}
                    k => map(x -> (x isa Union{BatchedReal,BatchedVal} ? batch_index(x,I) : x), v)
                elseif v isa AbstractArray{<:Union{BatchedReal,BatchedVal}}
                    k => batch_index.(v, I)
                else
                    k => v
                end
            end)
        end)
    end)
end

@doc doc"""
    unbatch(chains::Chains)

Expand each chain in this `Chains` object by unbatching it. 
"""
unbatch(chains::Chains) = Chains(mapreduce(unbatch, vcat, chains))



@doc doc"""
    mean_std_and_errors(samples; N_bootstrap=10000)

Get the mean and standard deviation of a set of correlated `samples` from a
chain where the error on the mean and standard deviation is estimated with
bootstrap resampling using the calculated "effective sample size" of the chain.
"""
function mean_std_and_errors(samples; N_bootstrap=10000, N_in_paren=2, tol=50)
    
    Neff = round(Int, length(samples) / @ondemand(PyCall.pyimport)(:emcee).autocorr.integrated_time(samples; tol)[1])
    
    μ = mean(samples)
    σ = std(samples)

    SEμ = std([mean(samples[rand(1:end, Neff)]) for i=1:N_bootstrap])
    SEσ = std([ std(samples[rand(1:end, Neff)]) for i=1:N_bootstrap])
    
    "$(paren_errors(μ, SEμ; N_in_paren=N_in_paren)) ± $(paren_errors(σ, SEσ; N_in_paren=N_in_paren))"
    
end

@doc doc"""
    paren_errors(μ, σ; N_in_paren=2)

Get a string represntation of `μ ± σ` in "parenthesis" format, e.g. `1.234 ±
0.012` becomes `1.234(12)`.
"""
function paren_errors(μ, σ; N_in_paren=2)
    N = round(Int, floor(log10(1/σ))) + N_in_paren
    fmt = "%.$(N)f"
    @ondemand(Formatting.sprintf1)(fmt, μ)*"($(round(Int,σ*10^N)))"
end



# an N-dimensional getdist Kernel Density Estimate
struct GetDistKDE{N}
    kde
end
(k::GetDistKDE)(x...) = k.kde(x...)

"""
    kde(samples::AbstractVector; [boundary=(min,max), normalize="integral" or "max"])
    kde(samples::AbstractMatrix; [boundary=[(min1,max1),(min2,max2)], normalize="integral" or "max", smooth_scale_2D])

Return a Kernel Density Estimate for a set of 1D or 2D samples. The
return object is a function which can be evaluated anywhere to compute
the PDF. If provided, `boundary` specifies a hard upper/lower bound
for the 1 or 2 or parameters, `normalize` specifies whether to
normalize the PDF to unit integral or unit maximum, and
`smooth_scale_2D` specifies how much smoothing to do for the 2D case.

Based on Python [GetDist](https://getdist.readthedocs.io/en/latest/intro.html), 
which must be installed.
"""
function kde(samples::AbstractVector; boundary=(nothing,nothing), normalize="integral")
    getdist = @ondemand(PyCall.pyimport)("getdist")
    getdist.chains.print_load_details = false
    kde = getdist.MCSamples(;
        samples, weights=nothing, names=["x"], ranges=Dict("x"=>boundary)
    )
    GetDistKDE{1}(kde.get1DDensity(0).normalize(normalize))
end

function kde(samples::AbstractMatrix; boundary=((nothing,nothing),(nothing,nothing)), normalize="integral", smooth_scale_2D=nothing)
    if size(samples,1) == 2
        samples = samples'
    elseif size(samples,2) != 2
        error("KDE only supports 1 or 2 dimensional samples.")
    end
    getdist = @ondemand(PyCall.pyimport)("getdist")
    getdist.chains.print_load_details = false
    kde = getdist.MCSamples(;
        samples, weights=nothing, names=["x","y"], ranges=Dict("x"=>boundary[1], "y"=>boundary[2])
    )
    density_kwargs = isnothing(smooth_scale_2D) ? () : (;smooth_scale_2D)
    GetDistKDE{2}(kde.get2DDensity(0, 1; density_kwargs...).normalize(normalize))
end