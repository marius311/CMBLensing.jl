export plot

pretty_name(s::Symbol) = pretty_name(string(s))
pretty_name(s::String) = s[1:1]*" "*(Dict('x'=>"map",'l'=>"fourier")[s[2]])

# generic plotting some components of a FlatField
function plot(f::FlatField{T,P}, axs, which; kwargs...) where {T,Θ,N,P<:Flat{Θ,N}}
    for (ax,k) in zip(axs,which)
        ax = plot(f[k]; ax=ax, title="$(pretty_name(k)) ($(N)x$(N) @ $(Θ)')", kwargs...)
    end
end

# plotting a real matrix
function plot(m::AbstractMatrix{<:Real}; title=nothing, kwargs...)
    m[isinf(m)]=NaN
    ax = pyimport(:seaborn)[:heatmap](m; mask=isnan(m), xticklabels=false, yticklabels=false, square=true, kwargs...)
    title!=nothing && ax[:set_title](title)
    ax
end

# plotting a complex matrix 
# we assume its a ~N×2N matrix (like a real FFT), and create a new ~2N×2N matrix
# with the real part on the upper half and the imaginary part mirrored on the
# bottom, with a row of NaN's inbetween to visually separate
function plot(m::AbstractMatrix{Complex{T}}; kwargs...) where {T}
    plot([real(m); fill(NaN,size(m,2))'; imag(m[end:-1:1,:])]; kwargs...)
end

# FlatS0
function plot(fs::AbstractMatrix{<:FlatS0}; plotsize=4, which=[:Tx], kwargs...)
    (length(which)==1) || throw(ArgumentError("Can't plot matrix of FlatS0's with multiple components, $(which)"))
    fig,axs = subplots(size(fs,1), size(fs,2); figsize=plotsize.*(size(fs,2),size(fs,1)), squeeze=false)
    for i=eachindex(fs)
        plot(fs[i], [axs[i]], which; kwargs...)
    end
end
function plot(fs::AbstractVector{<:FlatS0}; plotsize=4, which=[:Tx], kwargs...)
    fig,axs = subplots(length(fs), length(which); figsize=plotsize.*(length(which),length(fs)), squeeze=false)
    for i=1:length(fs)
        plot(fs[i], axs[i,:], which; kwargs...)
    end
end
plot(f::FlatS0; kwargs...) = plot([f]; kwargs...)

# FlatS2
function plot(fs::AbstractVector{<:FlatS2}; plotsize=4, which=[:Ex,:Bx], kwargs...)
    ncol = length(which)
    fig,axs = subplots(length(fs),ncol; figsize=(plotsize.*(ncol,length(fs))), squeeze=false)
    for i=1:length(fs)
        plot(fs[i], axs[i,:], which; kwargs...)
    end
end
plot(f::FlatS2; kwargs...) = plot([f]; kwargs...)

# FieldTuple{<:FlatS0,<:FlatS2} (i.e., TEB)
function plot(f::Field2Tuple{<:FlatS0{T,P},<:FlatS2{T,P}}, axs, which; kwargs...) where {T,Θ,N,P<:Flat{Θ,N}}
    for (ax,k) in zip(axs,which)
        m = string(k)[1]=='T' ? f.f1[k] : f.f2[k]
        ax = plot(m; ax=ax, title="$(pretty_name(k)) ($(N)x$(N) @ $(Θ)')", kwargs...)
    end
end
function plot(fs::AbstractVector{<:Field2Tuple{<:FlatS0,<:FlatS2}}; plotsize=4, which=[:Tx,:Ex,:Bx], kwargs...)
    fig,axs = subplots(length(fs),length(which); figsize=(plotsize.*(length(which),length(fs))), squeeze=false)
    for i=1:length(fs)
        plot(fs[i], axs[i,:], which; kwargs...)
    end
end
plot(f::Field2Tuple{<:FlatS0,<:FlatS2}; kwargs...) = plot([f]; kwargs...)
