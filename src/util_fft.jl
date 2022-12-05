
@doc """
The number of threads used by FFTW for CPU FFTs (default is the environment
variable `FFTW_NUM_THREADS`, or if that is not specified its
`Sys.CPU_THREADS÷2`). This must be set before creating any `FlatField` objects.
"""
FFTW_NUM_THREADS = 1
@init global FFTW_NUM_THREADS = parse(Int,get(ENV,"FFTW_NUM_THREADS","$(Sys.CPU_THREADS÷2)"))


@doc """
Time-limit for FFT planning on CPU (default: 5 seconds). This must be set before
creating any `FlatField` objects.
"""
FFTW_TIMELIMIT = 5


# a set of wrapper FFT functions which use a @memoize'd plan
_fft_arr_type(arr) = basetype(typeof(parent(arr)))
m_rfft(arr::AbstractArray{T,N}, dims) where {T<:Real,N} = m_plan_rfft(_fft_arr_type(arr){T,N}, dims, size(arr)...) * arr
function m_irfft(arr::AbstractArray{T,N}, d, dims) where {T,N}
    output_size = size(arr)
    @set! output_size[first(dims)] = d
    m_plan_rfft(_fft_arr_type(arr){real(T),N}, dims, output_size...) \ arr
end
m_rfft!(dst, arr::AbstractArray{T,N}, dims) where {T<:Real,N} = mul!(dst, m_plan_rfft(_fft_arr_type(arr){T,N}, dims, size(arr)...), arr)
m_irfft!(dst, arr::AbstractArray{T,N}, dims) where {T,N} = ldiv!(dst, m_plan_rfft(_fft_arr_type(arr){real(T),N}, dims, size(dst)...), copy_if_fftw(arr))
m_fft(arr::AbstractArray{T,N}, dims) where {T,N} = m_plan_fft(_fft_arr_type(arr){complex(T),N}, dims, size(arr)...) * arr
m_ifft(arr::AbstractArray{T,N}, dims) where {T,N} = m_plan_fft(_fft_arr_type(arr){complex(T),N}, dims, size(arr)...) \ arr
m_fft!(dst, arr::AbstractArray{T,N}, dims) where {T,N} = mul!(dst, m_plan_fft(_fft_arr_type(arr){complex(T),N}, dims, size(arr)...), complex(arr))
m_ifft!(dst, arr::AbstractArray{T,N}, dims) where {T,N} = ldiv!(dst, m_plan_fft(_fft_arr_type(arr){complex(T),N}, dims, size(arr)...), complex(arr))
@memoize function m_plan_rfft(::Type{A}, dims::Dims, sz...) where {T, N, A<:AbstractArray{T,N}, Dims}
    FFTW.set_num_threads(FFTW_NUM_THREADS)
    plan_rfft(A(undef, sz...), dims; (A <: Array ? (timelimit=FFTW_TIMELIMIT,) : ())...)
end
@memoize function m_plan_fft(::Type{A}, dims::Dims, sz...) where {T, N, A<:AbstractArray{T,N}, Dims}
    FFTW.set_num_threads(FFTW_NUM_THREADS)
    plan_fft(A(undef, sz...), dims; (A <: Array ? (timelimit=FFTW_TIMELIMIT,) : ())...)
end
Zygote.@nograd m_plan_fft, m_plan_rfft
# FFTW (but not MKL) destroys the input array for inplace inverse real
# FFTs, so we need a copy. see https://github.com/JuliaMath/FFTW.jl/issues/158
copy_if_fftw(x) = (x isa Array && FFTW.fftw_provider == "fftw") ? copy(x) : x


"""
Convert a matrix A which is the output of a real FFT to a real vector, keeping
only unqiue real/imaginary entries of A
"""
function rfft2vec(A::AbstractMatrix)
    m,n = size(A)
    ireal,iimag = fftsyms(Val(m),Val(n))
    [real(A[ireal]); imag(A[iimag])]
end

"""
Convert a vector produced by rfft2vec back into a complex matrix.
"""
function vec2rfft(v::AbstractVector{<:Real})
    n = round(Int,sqrt(length(v)))
    m = n÷2+1
    nreal = (n^2)÷2 + (iseven(n) ? 2 : 1)
    ireal,iimag,inegks = fftsyms(Val(m),Val(n))
    A = fill(NaN+im*NaN,m,n)
    A[ireal] = v[1:nreal]
    A[iimag] += im*v[nreal+1:end]
    for i=1:m, j=1:n
        if isnan(A[i,j])
            A[i,j] = A[inegks[i,j]...]'
        end
    end
    A
end


"""
Convert an M×N matrix (with M=N÷2+1) which is the output a real FFT to a full
N×N one via symmetries.
"""
unfold(Tls::AbstractArray{<:Any,3}, Ny) = mapslices(X -> unfold(X, Ny), Array(Tls), dims=(1,2))
unfold(Tl::AbstractMatrix{<:Any}, Ny) = unfold(Array(Tl), Ny)
function unfold(Tl::Matrix{<:Any}, Ny::Int)
    m,n = size(Tl)
    @assert m==Ny÷2+1
    m2 = iseven(Ny) ? 2m : 2m+1
    n2 = iseven(n) ?  n+2 : n+3
    Tlu = similar(Tl,Ny,n)
    Tlu[1:m,1:n] = Tl
    @inbounds for i=m+1:Ny
        Tlu[i,1] = Tl[m2-i, 1]'
        @simd for j=2:n
            Tlu[i,j] = Tl[m2-i, n2-j]'
        end
    end
    Tlu
end



"""
Arguments `m` and `n` refer to the sizes of an `m`×`n` matrix (call it `A`) that is the output of a
real FFT (thus `m=n÷2+1`)

Returns a tuple of (ireal, iimag, negks) where these are

* `ireal` — `m`×`n` mask corrsponding to unique real entries of `A`
* `iimag` — `m`×`n` mask corrsponding to unique imaginary entries of `A`
* `negks` — `m`×`n` matrix of giving the index into A where the negative k-vector
            is, s.t. `A[i,j] = A[negks[i,j]]'`
"""
@generated function fftsyms(::Val{m},::Val{n}) where {m,n}
    k = ifftshift(-n÷2:(n-1)÷2)
    ks = tuple.(k',k)[1:n÷2+1,:]
    wrapk(k) = mod(k+n÷2,n) - n÷2
    negk(k) = @. wrapk(-k)
    k_in_ks(k) = -n÷2<=k[1]<=(n-1)÷2 && (0<=k[2]<=(n-1)÷2 || k[2]==-n÷2)
    ireal = map(k->(negk(k)==k || !k_in_ks(negk(k)) || (k[1]>0 || k[2]>0)), ks)
    iimag = map(k->(negk(k)!=k && !k_in_ks(negk(k)) || (k[1]>0 || k[2]>0)), ks)
    indexof(k) = (mod(k[2],n)+1, mod(k[1],n)+1)
    inegks = indexof.(negk.(ks))
    inegks[.!k_in_ks.(negk.(ks))] .= Ref((0,0))
    ireal,iimag,inegks#,ks,negk.(ks)#,k_in_ks.(negk.(ks)),map(k->k_in_ks(negk(k)),ks)
end


"""
    rfft_degeneracy_fac(n)
    
Returns an Array which is 2 if the complex conjugate of the
corresponding entry in the half-plane real FFT appears in the
full-plane FFT, and is 1 othewise. `n` is the length of the first
dimension of the full-plane FFT. The following identity holds:

    sum(abs2.(fft(x)) = sum(rfft_degeneracy_fac(size(x,1)) .* abs2.(rfft(x))
"""
function rfft_degeneracy_fac(n)
    if iseven(n)
        [1; fill(2,n÷2-1); 1]
    else
        [1; fill(2,n÷2)]
    end
end 

# this switched names & return type somewhere near FFTW 1.3 so need this for backwards compatibility
fftw_provider() = isdefined(FFTW, :fftw_vendor) ? string(FFTW.fftw_vendor) : FFTW.fftw_provider