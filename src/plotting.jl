using PyPlot
import PyPlot: plot

export plot

pretty_name(s::Symbol) = pretty_name(string(s))
pretty_name(s::String) = s[1:1]*" "*(Dict('x'=>"map",'l'=>"fourier")[s[2]])

# generic plotting some components of a FlatField
function plot(f::FlatField{T,P}, axs, which; units=:deg, ticklabels=true, kwargs...) where {T,Θ,N,P<:Flat{Θ,N}}
    x = Θ*N/Dict(:deg=>60,:arcmin=>1)[units]/2
    extent = [-x,x,-x,x]
    for (ax,k) in zip(axs,which)
        plot(f[k]; ax=ax, extent=extent, title="$(pretty_name(k)) ($(N)x$(N) @ $(Θ)')", kwargs...)
        if ticklabels
            @pydef type MyFmt <: pyimport(:matplotlib)[:ticker][:ScalarFormatter]
                __call__(self,v,p=nothing) = py"super"(MyFmt,self)[:__call__](v,p)*Dict(:deg=>"°",:arcmin=>"′")[units]
            end
            ax[:xaxis][:set_major_formatter](MyFmt())
            ax[:yaxis][:set_major_formatter](MyFmt())
            ax[:set_ylabel]("Dec")
            ax[:set_xlabel]("RA")
            ax[:tick_params](labeltop=false, labelbottom=true)
        else
            ax[:tick_params](labeltop=false, labelleft=false)
        end
    end
end

# plotting a real matrix
function plot(m::AbstractMatrix{<:Real}; ax=gca(), title=nothing, vlim=:sym, cmap="RdBu_r", cbar=true, kwargs...)
    
    # some logic to automatically get upper/lower limits
    if vlim==:sym
        vmax = quantile(abs.(m[!isnan.(m)][:]),0.999)
        vmin = -vmax
    elseif vlim==:asym
        vmin, vmax = (quantile(m[!isnan.(m)][:],q) for q=(0.001,0.999))
    elseif isa(vlim,Tuple)
        vmin, vmax = vlim
    else
        vmax = vlim
        vmin = -vmax
    end
       
    m[isinf(m)]=NaN
    
    cax = ax[:matshow](m; vmin=vmin, vmax=vmax, cmap=cmap, rasterized=true, kwargs...)
    cbar && gcf()[:colorbar](cax,ax=ax)
    title!=nothing && ax[:set_title](title, y=1)
    ax
end

# plotting a complex matrix 
# we assume its a ~N×2N matrix (like a real FFT), and create a new ~2N×2N matrix
# with the real part on the upper half and the imaginary part mirrored on the
# bottom, with a row of NaN's inbetween to visually separate
function plot(m::AbstractMatrix{Complex{T}}; kwargs...) where {T}
    plot(log10.(abs.([real(m); fill(NaN,size(m,2))'; imag(m[end:-1:1,:])])); kwargs...)
end

# FlatS0
function plot(fs::AbstractMatrix{<:FlatS0}; plotsize=4.8, which=[:Tx], kwargs...)
    (length(which)==1) || throw(ArgumentError("Can't plot matrix of FlatS0's with multiple components, $(which)"))
    fig,axs = subplots(size(fs)...; figsize=plotsize.*[1.4*size(fs,2),size(fs,1)], squeeze=false)
    for i=eachindex(fs)
        plot(fs[i], [axs[i]], which; kwargs...)
    end
    fig,axs
end
function plot(fs::AbstractVector{<:FlatS0}; plotsize=4, which=[:Tx], kwargs...)
    fig,axs = subplots(length(fs), length(which); figsize=plotsize.*(length(which),length(fs)), squeeze=false)
    for i=1:length(fs)
        plot(fs[i], axs[i,:], which; kwargs...)
    end
    fig,axs
end
plot(f::FlatS0; kwargs...) = plot([f]; kwargs...)

# FlatS2
function plot(fs::AbstractVector{<:FlatS2}; plotsize=4.8, which=[:Ex,:Bx], kwargs...)
    ncol = length(which)
    fig,axs = subplots(length(fs),ncol; figsize=(plotsize.*[1.4ncol,length(fs)]), squeeze=false)
    for i=1:length(fs)
        plot(fs[i], axs[i,:], which; kwargs...)
    end
    fig,axs
end
function plot(fs::RowVector{<:FlatS2}; plotsize=4.8, which=[:Ex,:Bx], kwargs...)
    ncol = length(which)
    fig,axs = subplots(ncol,length(fs); figsize=(plotsize.*[1.4length(fs),ncol]), squeeze=false)
    for i=1:length(fs)
        plot(fs[i], axs[:,i], which; kwargs...)
    end
    fig,axs
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
    fig,axs
end
function plot(fs::RowVector{<:Field2Tuple{<:FlatS0,<:FlatS2}}; plotsize=4.8, which=[:Tx,:Ex,:Bx], kwargs...)
    fig,axs = subplots(length(which), length(fs); figsize=(plotsize.*[1.4length(fs),length(which)]), squeeze=false)
    for i=1:length(fs)
        plot(fs[i], axs[:,i], which; kwargs...)
    end
    fig,axs
end
plot(f::Field2Tuple{<:FlatS0,<:FlatS2}; kwargs...) = plot([f]; kwargs...)



function animate(fields::Vector{<:FlatS0{T,P}}, interval=50, units=:deg) where {T,Θ,N,P<:Flat{Θ,N}}
    l = Θ*N/Dict(:deg=>60,:arcmin=>1)[units]/2
    img = imshow(fields[1][:Tx],cmap="RdBu_r",extent=(-l,l,-l,l))
    ax = gca()
    @pydef type MyFmt <: pyimport(:matplotlib)[:ticker][:ScalarFormatter]
        __call__(self,v,p=nothing) = py"super"(MyFmt,self)[:__call__](v,p)*"°"
    end
    ax[:xaxis][:set_major_formatter](MyFmt())
    ax[:yaxis][:set_major_formatter](MyFmt())
    ax[:set_ylabel]("Dec")
    ax[:set_xlabel]("RA")
    ax[:tick_params](labeltop=false, labelbottom=true)
    
    ani = pyimport("matplotlib.animation")[:FuncAnimation](gcf(), 
        i->(img[:set_data](fields[i][:Tx]);(img,)), 
        1:length(fields),
        interval=interval, blit=true
        )
    close()
    HTML(ani[:to_html5_video]())
end
