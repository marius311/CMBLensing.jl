
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
unfold(Tl::AbstractMatrix) = unfold(Array(Tl))
function unfold(Tl::Matrix{Complex{T}}) where {T}
    m,n = size(Tl)
    @assert m==n÷2+1
    m2 = iseven(n) ? 2m : 2m+1
    Tlu = Array{Complex{T}}(undef,n,n)
    Tlu[1:m,1:n] = Tl
    @inbounds for i=m+1:n
        Tlu[i,1] = Tl[m2-i, 1]'
        @simd for j=2:n
            Tlu[i,j] = Tl[m2-i, m2-j]'
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
