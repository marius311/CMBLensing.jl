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
* `unbatch` — If true, run `unbatch` on the chains if they are batched.
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
    # Keep only batch widths and map presence; dropmaps must release fields
    # before thinning and unbatching.
    dropped_batch_hints = IdDict{Any,Int}()
    dropped_maps = IdDict{Any,Nothing}()
    chains = jldopen(filename) do io
        ks = keys(io)
        chunk_ks = sort([k for k in ks if startswith(k,"chunks_")], by=k->parse(Int,k[8:end]))
        chunk_ks = chunk_ks[burnin_chunks>=0 ? (burnin_chunks+1:end) : (end+burnin_chunks+1:end)]
        chains = [[]]
        @showprogress (progress ? 1 : Inf) for (i,k) in enumerate(chunk_ks)
            chunk = read(io,k)
            if i == 1
                chains = chunk
            else
                length(chunk) == length(chains) || throw(ArgumentError(
                    "Chunk $k contains $(length(chunk)) chains, but previous chunks contain $(length(chains))"))
                append!.(chains, chunk)
            end
            if dropmaps
                for chain in chunk, sample in chain
                    if unbatch
                        marker = _sample_batch_marker(sample)
                        if !isnothing(marker)
                            dropped_batch_hints[sample] = batch_length(marker)
                        end
                    end
                    if thin == :hasmaps && get(sample, :ϕ, nothing) isa Field
                        dropped_maps[sample] = nothing
                    end
                    filter!(((_,v),)->!(v isa Field), sample)
                end
            end
        end
        chains
    end
    if thin isa Int
        chains = [chain[burnin>=0 ? ((1+burnin):thin:end) : (end+(1+burnin):thin:end)] for chain in chains]
    elseif thin == :hasmaps
        chains = [[samp for samp in chain[(1+burnin):end] if haskey(samp, :ϕ) || haskey(dropped_maps, samp)] for chain in chains]
    elseif thin isa Function
        chains = [filter(thin,chain) for chain in chains]
    else
        error("`thin` should be an Int, :hasmaps, or a filter function")
    end
    chains = wrap_chains(chains)
    if unbatch
        if dropmaps && !isempty(dropped_batch_hints)
            chains = Chains(mapreduce(chain -> _unbatch(chain, dropped_batch_hints), vcat, chains))
        else
            chains = CMBLensing.unbatch(chains)
        end
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
    if !isempty(c)
        _,cols = displaysize(io)
        for k in keys(c[end])
            str = string("$(indent)$(k) => ", repr(c[k]; context=(:limit => true)))
            println(io, _truncate_at_width_or_chars(str, cols))
        end
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
If a custom sample contains no batch-bearing field or value, its batch length
cannot be inferred from the sample and it is treated as one chain.
"""
unbatch(chain::Chain) = _unbatch(chain, nothing)

function _sample_batch_marker(sample)
    candidates = Any[]
    logpdf = get(sample, :logpdf, nothing)
    !(isnothing(logpdf) || ismissing(logpdf)) && push!(candidates, logpdf)
    ϕ = get(sample, :ϕ, nothing)
    ϕ isa Field && push!(candidates, ϕ)
    for v in values(sample)
        v isa Union{BatchedReal,BatchedVal} && push!(candidates, v)
        if v isa NamedTuple
            for component in values(v)
                component isa Union{BatchedReal,BatchedVal} && push!(candidates, component)
            end
        elseif v isa AbstractArray{<:Union{BatchedReal,BatchedVal}}
            append!(candidates, v)
        end
    end
    for v in values(sample)
        v isa Field && push!(candidates, v)
    end
    # A singleton may be shared, but all non-singleton batches must agree.
    marker = nothing
    width = 1
    for candidate in candidates
        applicable(batch_length, candidate) || continue
        n = batch_length(candidate)
        if n > 1
            (width == 1 || width == n) || throw(ArgumentError(
                "Sample contains incompatible batch lengths $width and $n"))
            width = n
            marker = candidate
        elseif isnothing(marker)
            marker = candidate
        end
    end
    marker
end

function _unbatch(chain::Chain, batch_hints)
    isempty(chain) && return [chain]
    D = 1
    for sample in chain
        marker = _sample_batch_marker(sample)
        marker_width = isnothing(marker) ? 1 : batch_length(marker)
        hint_width = isnothing(batch_hints) ? 1 : get(batch_hints, sample, 1)
        for n in (marker_width, hint_width)
            n <= 1 && continue
            (D == 1 || D == n) || throw(ArgumentError(
                "Chain contains incompatible batch lengths $D and $n"))
            D = n
        end
    end
    (D==1) && return [chain]
    Chains(map(1:D) do I
        Chain(map(chain) do samp
            Dict(map(collect(samp)) do (k,v)
                if v isa Union{BatchedReal,BatchedVal}
                    k => batch_index(v, batch_length(v) == 1 ? 1 : I)
                elseif v isa Union{FlatField,FieldTuple}
                    k => (batch_length(v) == 1 ? v : batch_index(v, I))
                elseif v isa NamedTuple
                    k => map(x -> (x isa Union{BatchedReal,BatchedVal} ?
                        batch_index(x, batch_length(x) == 1 ? 1 : I) : x), v)
                elseif v isa AbstractArray{<:Union{BatchedReal,BatchedVal}}
                    k => map(x -> batch_index(x, batch_length(x) == 1 ? 1 : I), v)
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
    
    Neff = round(Int, MCMCDiagnosticTools.ess(reshape(samples, :, 1, 1))[1])
    
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
function kde(samples::AbstractVector; boundary=(nothing,nothing), normalize="integral", smooth_scale_1D=nothing)
    getdist = pyimport("getdist")
    getdist.chains.print_load_details = false
    kde = getdist.MCSamples(;
        samples, weights=nothing, names=["x"], ranges=Dict("x"=>boundary)
    )
    density_kwargs = isnothing(smooth_scale_1D) ? () : (;smooth_scale_1D)
    GetDistKDE{1}(kde.get1DDensity(0; density_kwargs...).normalize(normalize))
end

function kde(samples::AbstractMatrix; boundary=((nothing,nothing),(nothing,nothing)), normalize="integral", smooth_scale_2D=nothing)
    if size(samples,1) == 2
        samples = samples'
    elseif size(samples,2) != 2
        error("KDE only supports 1 or 2 dimensional samples.")
    end
    getdist = pyimport("getdist")
    np = pyimport("numpy")
    getdist.chains.print_load_details = false
    kde = getdist.MCSamples(;
        samples=np.asarray(samples), weights=nothing, names=["x","y"], ranges=Dict("x"=>boundary[1], "y"=>boundary[2])
    )
    density_kwargs = isnothing(smooth_scale_2D) ? () : (;smooth_scale_2D)
    GetDistKDE{2}(kde.get2DDensity(0, 1; density_kwargs...).normalize(normalize))
end
