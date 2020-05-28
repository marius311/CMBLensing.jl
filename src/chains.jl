import Base: getindex, lastindex


@doc doc"""
    load_chains(filename; burnin=0, thin=1, join=false)
    
Load a single chain or multiple parallel chains which were written to a file by
[`sample_joint`](@ref). 

Keyword arguments: 

* `burnin` — Remove this many samples from the start of each chain.
* `thin` — If `thin` is an integer, thin the chain by this factor. If
  `thin == :hasmaps`, return only samples which have maps saved. If thin is a
  `Function`, filter the chain by this function (e.g. `thin=haskey(:g)` on Julia 1.5+)
* `unbatch` — If true, [unbatch](@ref) the chains if they are batched. 
* `join` — If true, concatenate all the chains together.
* `skip_missing_chunks` — Skip missing chunks in the chain instead of
  terminating the chain there. 


The object returned by this function is a `Chain` or `Chains` object, which
simply wraps an `Array` of `Dicts` or an `Array` of `Array` of `Dicts`,
respectively (each sample is a `Dict`). The wrapper object has some extra
indexing properties for convenience: 

* It can be indexed as if it were a single multidimensional object, e.g.
  `chains[1,:,:accept]` would return the `:accept` key of all samples in the
  first chain.
* Leading colons can be dropped, i.e. `chains[:,:,:accept]` is the same as
  `chains[:accept]`. 
* If some samples are missing a particular key, `missing` is returned for those
  samples insted of an error.
* The recursion goes arbitrarily deep into the objects it finds. E.g., since
  sampled parameters are stored in a `NamedTuple` like `(Aϕ=1.3,)` in the `θ`
  key of each sample `Dict`, you can do `chain[:θ,:Aϕ]` to get all `Aϕ` samples
  as a vector. 


"""
function load_chains(filename; burnin=0, thin=1, join=false, unbatch=true)
    chains = jldopen(filename) do io
        ks = keys(io)
        chunk_ks = [k for k in ks if startswith(k,"chunks_")]
        for (isfirst,k) in flagfirst(sort(chunk_ks, by=k->parse(Int,k[8:end])))
            if isfirst
                chains = read(io,k)
            else
                append!.(chains, read(io,k))
            end
        end
        chains
    end
    if thin isa Int
        chains = [chain[(1+burnin):thin:end] for chain in chains]
    elseif thin == :hasmaps
        chains = [[samp for samp in chain[(1+burnin):end] if :ϕ in keys(samp)] for chain in chains]
    elseif thin isa Function
        chains = [filter(thin,chain) for chain in chains]
    else
        error("`thin` should be an Int or :hasmaps")
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
getindex(c::Chains, k::Symbol, ks...) = _getindex(c.chains, :, :, k, ks...)
getindex(c::Chains, k,         ks...) = _getindex(c.chains, k, ks...)
size(c::Chains) = size(c.chains)
function Base.print_array(io::IO, cs::Chains; indent="  ")
    for c in cs
        print(io, "  ")
        Base.summary(io, c)
        println(io)
        Base.print_array(io, c, indent="    ")
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
    for k in keys(c[1])
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


wrap_chains(chains::Vector{<:Vector{<:Dict}}) = Chains(Chain.(chains))
wrap_chains(chain::Vector{<:Dict}) = Chain(chain)


# batching
@doc doc"""
    unbatch(chain::Chain)

Convert a chain of batch-length-`D` fields to `D` chains of unbatched fields. 
"""
function unbatch(chain::Chain)
    D = batchsize(chain[1][:ϕ°])
    (D==1) && return chain
    Chains(map(1:D) do I
        Chain(map(chain) do samp
            Dict(map(collect(samp)) do (k,v)
                if v isa Union{BatchedReal,FlatField}
                    k => batchindex(v, I)
                elseif v isa NamedTuple{<:Any, <:NTuple{<:Any,<:BatchedVals}}
                    k => map(x -> (x isa BatchedReal ? batchindex(x,I) : x), v)
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
unbatch(chains::Chains) = Chains(reduce(vcat, map(unbatch, chains)))