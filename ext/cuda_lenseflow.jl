# Fuse the pixelwise spin contraction rather than allocating a product and a
# reduction at every RK4 stage. Keep the generic broadcasting path for views,
# mismatched batch widths, and other layouts.
function CMBLensing._lambert_spin_mul!(dst::CuArray, a::CuArray{T}, b::CuArray{T}) where {T<:Union{Float32,Float64}}
    plane = size(a, 1) * size(a, 2)
    nspin = size(a, 3)
    if size(a) != size(b) || ndims(a) > 4 || nspin == 0 ||
       length(dst) * nspin != length(a) || size(dst, 1) * size(dst, 2) != plane
        return invoke(CMBLensing._lambert_spin_mul!, Tuple{Any,Any,Any}, dst, a, b)
    end
    isempty(dst) && return dst
    @cuda threads=256 blocks=cld(length(dst), 256) _spin_mul_kernel!(dst, a, b, plane, nspin)
    dst
end

function _spin_mul_kernel!(dst, a, b, plane, nspin)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(dst)
        pixel = (idx - 1) % plane + 1
        batch = (idx - 1) ÷ plane
        start = pixel + batch * plane * nspin
        @inbounds value = a[start] * b[start]
        for spin in 2:nspin
            @inbounds value += a[start + (spin-1)*plane] * b[start + (spin-1)*plane]
        end
        @inbounds dst[idx] = value
    end
    nothing
end

const CuCachedLenseFlow = CachedLenseFlow{S,T,D,LP,DP,LF,DF} where {
    S<:CMBLensing._GraphRK4Solver,T,D<:Diagonal,LP<:Field,DP<:Field,LF<:CuBaseField,DF<:Field}

function CMBLensing._inplace_rfft_plan(arr::CuArray{T,N}, dims, sz...) where {T,N}
    plans = get(task_local_storage(), :CMBLensing_graph_fft_plans, nothing)
    if isnothing(plans)
        return CMBLensing.m_plan_rfft(CMBLensing._fft_arr_type(arr){real(T),N}, dims, sz...)
    end
    plans[(real(eltype(arr)), dims, sz)]
end

mutable struct LenseFlowGraphs
    task::Task
    context::CUDA.CuContext
    stream::CUDA.CuStream
    # Three velocities, each in two directions: at most six retained graphs.
    entries::Dict{Tuple{DataType,Bool},Any}
end

function CMBLensing._lenseflow_prepare!(L::CachedLenseFlow{<:CMBLensing._GraphRK4Solver})
    state = L.odesolve.graphs.state
    if state isa LenseFlowGraphs &&
       (state.task !== current_task() || state.context != CUDA.context() || state.stream != CUDA.stream())
        # Order a sequential transfer before coefficients or scratch are
        # overwritten. Library handles can also differ between tasks/streams.
        CUDA.context!(state.context) do
            CUDA.synchronize(state.stream)
        end
        L.odesolve.graphs.state = nothing
    end
    nothing
end

function _graph_supported(L)
    L.odesolve.nsteps > 0 &&
    L.memŁf isa Union{CMBLensing.LambertMap,CMBLensing.LambertQUMap,CMBLensing.LambertIQUMap} &&
    L.memŁϕ isa CMBLensing.LambertMap && L.memŁϕ isa CuBaseField &&
    eltype(L.memŁf) <: Union{Float32,Float64} &&
    eltype(L.memŁϕ) <: Union{Float32,Float64} &&
    CUDA.context(L.memŁf.arr) == CUDA.context() &&
    CUDA.context(L.memŁϕ.arr) == CUDA.context() &&
    isdefined(CUDA, :capture) && isdefined(CUDA, :instantiate)
end

# Each integration owns its FFT plans and work areas: the globally memoized
# plans can be used by ordinary calls or another cache while a graph runs.
# Retain them explicitly because CUDA 5 replay does not retain Julia objects.
function _graph_fft_plans(L)
    plans = Dict{Any,Any}()
    for f in (L.memŁf, L.memŁϕ)
        key = (eltype(f.arr), (1,2), size(f.arr))
        if !haskey(plans, key)
            p = plan_rfft(similar(f.arr), (1,2))
            inv(p) # Initialize the inverse before capture; p retains it.
            plans[key] = p
        end
    end
    plans
end

function _capture_lenseflow(L, F!, y₀, (t₀,t₁))
    # y₀ can alias L's scratch. Preserve it BEFORE warming up the velocity,
    # which overwrites that scratch, and never refresh it on this first call.
    yin, y = copy(y₀), copy(y₀)
    work = ntuple(_ -> similar(y₀), 5)
    run = () -> begin
        y .= yin
        CMBLensing._rk4!(F!, y, work..., L.odesolve.nsteps, t₀, t₁)
    end
    plans = _graph_fft_plans(L)
    graph = task_local_storage(:CMBLensing_graph_fft_plans, plans) do
        run(); CUDA.synchronize(CUDA.stream())
        run(); CUDA.synchronize(CUDA.stream())
        CUDA.capture(run; throw_error=false)
    end
    # An unsupported capture can be declined by CUDA. The last warmup is an
    # ordinary, complete integration of the preserved input in that case.
    exec = isnothing(graph) ? nothing : CUDA.instantiate(graph)
    (; yin, y, work, F!, plans, graph, exec)
end

function _replay_lenseflow(entry, y₀)
    entry.yin .= y₀
    CUDA.launch(entry.exec)
    # Results (including retained gradients) must not alias reusable work.
    copy(entry.y)
end

function CMBLensing._flow_solve(L::CuCachedLenseFlow,
        v::Union{typeof(CMBLensing.velocity),typeof(CMBLensing.velocityᴴ),typeof(CMBLensing.negδvelocityᴴ)},
        f, ts)
    if !_graph_supported(L) ||
       !(ts == (L.t₀ => L.t₁) || ts == (L.t₁ => L.t₀))
        return L.odesolve(v(L, f)..., ts)
    end
    CMBLensing._lenseflow_prepare!(L)
    if isnothing(L.odesolve.graphs.state)
        L.odesolve.graphs.state = LenseFlowGraphs(current_task(), CUDA.context(), CUDA.stream(), Dict{Tuple{DataType,Bool},Any}())
    end
    state = L.odesolve.graphs.state::LenseFlowGraphs
    key = (typeof(v), first(ts) == L.t₀)
    if !haskey(state.entries, key)
        # Promoted AD tangents can create one-off caches. Do not pay capture
        # costs unless this particular integration workspace is reused.
        state.entries[key] = :warmup
        return L.odesolve(v(L, f)..., ts)
    end
    entry = state.entries[key]
    isnothing(entry) && return L.odesolve(v(L, f)..., ts)
    entry !== :warmup && return _replay_lenseflow(entry, CMBLensing._flow_initial_state(L, v, f))
    F!, y₀ = v(L, f)
    entry = _capture_lenseflow(L, F!, y₀, ts)
    if isnothing(entry.exec)
        state.entries[key] = nothing
    else
        state.entries[key] = entry
        CUDA.launch(entry.exec)
    end
    copy(entry.y)
end
