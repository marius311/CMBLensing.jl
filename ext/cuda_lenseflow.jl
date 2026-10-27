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

const CuCachedLenseFlow = CachedLenseFlow{S,T,D,LP,DP,LF} where {S<:CMBLensing._GraphRK4Solver,T,D,LP,DP,LF<:CuBaseField}

mutable struct LenseFlowGraphs
    task::Task
    context::CUDA.CuContext
    stream::CUDA.CuStream
    # Three velocities, each in two directions: at most six retained graphs.
    entries::Dict{Tuple{DataType,Bool},Any}
end

function CMBLensing._lenseflow_prepare!(L::CuCachedLenseFlow)
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

# Retain the plans as well as their buffers. In CUDA 5 graph launches do not
# keep Julia objects alive, and FFT work areas must outlive every replay.
function _graph_fft_plans(L)
    map((L.memŁf, L.memŁϕ)) do f
        A = CMBLensing._fft_arr_type(f.arr)
        p = CMBLensing.m_plan_rfft(A{eltype(f.arr),ndims(f.arr)}, (1,2), size(f.arr)...)
        (p, inv(p))
    end
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
    run(); CUDA.synchronize(CUDA.stream())
    run(); CUDA.synchronize(CUDA.stream())
    graph = CUDA.capture(run; throw_error=false)
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
    if haskey(state.entries, key)
        entry = state.entries[key]
        return isnothing(entry) ? L.odesolve(v(L, f)..., ts) : _replay_lenseflow(entry, CMBLensing._flow_initial_state(L, v, f))
    end
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
