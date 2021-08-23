
using .PyPlot
using .PyPlot.PyCall
import .PyPlot: loglog, plot, semilogx, semilogy, figure, fill_between, imshow


### overloaded 1D plotting

for plot in (:plot, :loglog, :semilogx, :semilogy)

	# Cℓs
    @eval function ($plot)(ic::InterpolatedCℓs, args...; kwargs...)
		($plot)(ic.ℓ, ic.Cℓ, args...; kwargs...)
	end
	@eval function ($plot)(ic::NamedTuple{<:Any,<:NTuple{<:Any,<:InterpolatedCℓs}}, args...; kwargs...)
		($plot).(values(ic), args...; kwargs...)
	end
	@eval function ($plot)(ic::InterpolatedCℓs{<:Measurement}, args...; kwargs...)
		errorbar(ic.ℓ, Measurements.value.(ic.Cℓ), Measurements.uncertainty.(ic.Cℓ), args...; marker=".", ls="", capsize=2, kwargs...)
		($plot) in [:loglog,:semilogx] && xscale("log")
		($plot) in [:loglog,:semilogy] && yscale("log")
	end

	# Loess-interpolated
	@eval function ($plot)(f::Function, m::Loess.LoessModel, args...; kwargs...)
	    l, = ($plot)(m.xs, f.(m.ys), ".", args...; kwargs...)
        xs′ = vcat(map(1:length(m.xs)-1) do i
		    collect(range(m.xs[i],m.xs[i+1],length=10))[1:end-1]
		end..., [last(m.xs)])
	    ($plot)(xs′, f.(m.(xs′)), args...; c=l.get_color(), kwargs...)
	end
	@eval ($plot)(m::Loess.LoessModel, args...; kwargs...) = ($plot)(identity, m, args...; kwargs...)

	# 1D KDE
	@eval function ($plot)(f::Function, k::GetDistKDE{1}, args...; kwargs...)
	    ($plot)(k.kde.x, f.(k.kde.P), args...; kwargs...)
	end
	@eval ($plot)(k::GetDistKDE{1}, args...; kwargs...) = ($plot)(identity, k, args...; kwargs...)

end

# 2D KDE
function plot(k::GetDistKDE{2}, args...; color=nothing, kwargs...)
	@unpack colors = pyimport("matplotlib")
	args = k.kde.x, k.kde.y, k.kde.P, [k.kde.getContourLevels([0.95,0.68]); Inf]
	if color == nothing
		color = gca()._get_lines.get_next_color()
	end
	contourf(args...; colors=[(colors.to_rgb(color)..., α) for α in (0.4, 0.8)], kwargs...)
	contour(args...;  colors=color, kwargs...)
end

# Cℓ band
function fill_between(ic::InterpolatedCℓs{<:Measurement}, args...; kwargs...)
	fill_between(
		ic.ℓ, 
		((@. Measurements.value(ic.Cℓ) - x * Measurements.uncertainty(ic.Cℓ)) for x in (-1,1))...,
		args...; kwargs...
	)
end

### plotting CartesianFields

pretty_name(s) = pretty_name(Val.(Symbol.(split(string(s),"")))...)
pretty_name(::Val{s}, b::Val) where {s} = "$s "*pretty_name(b)
pretty_name(::Val{:x}) = "Map"
pretty_name(::Val{:l}) = "Fourier"

function _plot(f, ax, k, title, vlim, vscale, cmap; cbar=true, units=:deg, ticklabels=true, axeslabels=false, kwargs...)
    
	@unpack Nx, Ny = fieldinfo(f)
	ismap = endswith(string(k), "x")
	
	# default values
	if title == nothing
		if f isa FlatS0
			title = pretty_name(string(k)[2])
		else
			title = pretty_name(k)
		end
		if f isa LambertField
			title *= " ($(Ny)x$(Nx) @ $(f.θpix)')"
		elseif f isa EquiRectField
			title *= " ($(Ny)x$(Nx))"
		end
	end
	if vlim == nothing 
		vlim = ismap ? :sym : :asym
	end
	if vscale == nothing
		vscale = ismap ? :linear : :log
	end
	if cmap == nothing
		if ismap
			cmap = get_cmap("RdBu_r")
		else
			cmap = get_cmap("viridis")
			cmap.set_bad("lightgray")
		end
	end

	# build array
	if ismap
		arr = Array(f[k])
	else
		arr = abs.(ifftshift(unfold(Array(f[k]), Ny)))
	end
	if vscale == :log
		arr[arr .== 0] .= NaN
	end
	
	# auto vlim's
	if vlim==:sym
        vmax = quantile(abs.(arr[@. !isnan(arr)][:]),0.999)
        vmin = -vmax
    elseif vlim==:asym
        vmin, vmax = (quantile(arr[@. !isnan(arr)][:],q) for q=(0.001,0.999))
    elseif isa(vlim,Tuple)
        vmin, vmax = vlim
    else
        vmax = vlim
        vmin = -vmax
    end
	
	# make the plot
	if ismap
		if f isa LambertField
			extent = [-Nx,Nx,-Ny,Ny] .* f.θpix / 2 / Dict(:deg=>60,:arcmin=>1)[units]
		elseif f isa EquiRectField
			extent = rad2deg.([f.φspan..., f.θspan...]) .* Dict(:deg=>1,:arcmin=>60)[units]
		end
	else
		extent = [-1,1,-1,1] .* fieldinfo(f).nyquist
	end
	norm = vscale == :log ? matplotlib.colors.LogNorm() : nothing
	cax = ax.matshow(
		clamp.(arr, vmin, vmax); 
		vmin=vmin, vmax=vmax, extent=extent,
		cmap=cmap, rasterized=true, norm=norm,
		kwargs...
	)
	
	# annonate
    if cbar
		colorbar(cax,ax=ax,pad=0.01)
	end
    ax.set_title(title, y=1)
    if ticklabels
		if ismap
			@pydef mutable struct MyFmt <: pyimport(:matplotlib).ticker.ScalarFormatter
				__call__(self,v,p=nothing) = py"super"(MyFmt,self).__call__(v,p)*Dict(:deg=>"°",:arcmin=>"′")[units]
			end
			ax.xaxis.set_major_formatter(MyFmt())
			ax.yaxis.set_major_formatter(MyFmt())
			if axeslabels
				if f isa LambertField
					ax.set_xlabel("x")
					ax.set_ylabel("y")
				elseif f isa EquiRectField
					ax.set_xlabel(L"\phi")
					ax.set_ylabel(L"\theta")
				end
			end
		else
	        ax.set_xlabel(raw"$\ell_x$")
	        ax.set_ylabel(raw"$\ell_y$")
	        ax.tick_params(axis="x", rotation=45)
		end
        ax.tick_params(labeltop=false, labelbottom=true)
    else
        ax.tick_params(labeltop=false, labelleft=false)
    end
	
end


@doc doc"""
    plot(f::Field; kwargs...)
    plot(fs::VecOrMat{\<:Field}; kwarg...)
    
Plotting fields. 
"""
plot(f::Field; kwargs...) = plot([f]; kwargs...)
function plot(D::DiagOp; kwargs...)
	props = _sub_components[findfirst(((k,v),)->diag(D) isa @eval($k), _sub_components)][2]
	plot(
		[diag(D)]; 
		which = permutedims([Symbol(p) for (p,_) in props if endswith(p,r"[lx]")]),
		kwargs...
	)
end

function plot(
	fs :: AbstractVecOrMat{F}; 
	plotsize = 4,
	which = default_which(fs), 
	title = nothing, 
	vlim = nothing, 
	vscale = nothing,
	cmap = nothing,
	return_all = false, 
	kwargs...
) where {F<:CartesianField}
	
    (m,n) = size(tuple.(fs, which)[:,:])
    figsize = plotsize .* [1.4 * n * fs[1].Nx / fs[1].Ny, m]
	fig,axs = subplots(m, n; figsize, squeeze=false)
    axs = getindex.(Ref(axs), 1:m, (1:n)') # see https://github.com/JuliaPy/PyCall.jl/pull/487#issuecomment-456998345
    _plot.(fs,axs,which,title,vlim,vscale,cmap; kwargs...)
	
	if return_all
		(fig, axs, which)
	elseif isdefined(Main,:IJulia) && Main.IJulia.inited
		nothing # on IJulia, returning the figure can lead to it getting displayed twice
	else
		fig # returning the figure works on Juno/VSCode/Pluto
	end
	
end

default_which(::AbstractVecOrMat{<:CartesianS0})  = [:Ix]
default_which(::AbstractVecOrMat{<:CartesianS2})  = [:Ex :Bx]
default_which(::AbstractVecOrMat{<:EquiRectS2})   = [:Qx :Ux]
default_which(::AbstractVecOrMat{<:CartesianS02}) = [:Ix :Ex :Bx]
function default_which(fs::AbstractVecOrMat{<:CartesianField})
    try
        ensuresame((default_which([f]) for f in fs)...)
    catch x
        x isa AssertionError ? throw(ArgumentError("Must specify `which` argument by hand for plotting this combination of fields.")) : rethrow()
    end
end


### animations of CartesianFields

@doc doc"""
    animate(fields::Vector{\<:Vector{\<:Field}}; interval=50, motionblur=false, kwargs...)

"""
animate(f::AbstractVecOrMat{<:CartesianField}; kwargs...) = animate([f]; kwargs...)
animate(annonate::Function, args...; kwargs...) = animate(args...; annonate=annonate, kwargs...)
function animate(fields::AbstractVecOrMat{<:AbstractVecOrMat{<:CartesianField}}; fps=25, motionblur=false, annonate=nothing, filename=nothing, kwargs...)
    fig, axs, which = plot(first.(fields); return_all=true, kwargs...)
    motionblur = (motionblur == true) ? [0.1, 0.5, 1, 0.5, 0.1] : (motionblur == false) ? [1] : motionblur
    
    if (annonate!=nothing); annonate(fig,axs,which); end
    
    ani = pyimport("matplotlib.animation").FuncAnimation(fig, 
        i->begin
            for (f,ax,k) in tuple.(fields,axs,which)
                if length(f)>1
                    img = ax.images[1]
                    img.set_data(sum(x*getindex(f[mod1(i-j+1,length(f))],k) for (j,x) in enumerate(motionblur)) / sum(motionblur))
                end
            end
            first.(getproperty.(axs,:images))[:]
        end, 
        1:maximum(length.(fields)[:]),
        interval=1000/fps, blit=true
        )
    close()
    if filename!=nothing
        ani.save(filename,writer="imagemagick",savefig_kwargs=Dict(:facecolor=>fig.get_facecolor()))
        if endswith(filename,".gif")
            run(`convert -layers Optimize $filename $filename`)
        end
    end
    HTML(ani.to_html5_video())
end



### plotting HealpixFields

function plot(f::HealpixMap; kwargs...)
	hp.projview(
		collect(f.arr);
		cmap                  = "RdBu_r", 
		graticule             = true,
		graticule_labels      = true,
		xlabel                = L"\phi",
		ylabel                = L"\theta",
		custom_ytick_labels   = ["$(θ)°" for θ in 150:-30:30],
		latitude_grid_spacing = 30,
		cb_orientation        = "vertical",
		projection_type       = "mollweide",
		kwargs...
	)
end





### convenience
# for plotting in environments that only show a plot if its the last thing returned

function figure(plotfn::Function, args...; kwargs...)
	figure(args...; kwargs...)
	plotfn()
	gcf()
end


### chains

"""
    plot_kde(samples; [boundary, normalize, smooth_scale_2D], kwargs...)

Plot a Kernel Density Estimate PDF for a set of 1D or 2D samples.
`boundary`, `normalize`, and `smooth_scale_2D` keyword arguments are
passed to to the underlying call to [`kde`](@ref), and all others are
passed to the underlying call to `plot`.

Based on Python [GetDist](https://getdist.readthedocs.io/en/latest/intro.html), 
which must be installed. 
"""
function plot_kde(samples; boundary=nothing, normalize=nothing, smooth_scale_2D=nothing, kwargs...)
	kde_kwargs = filter(!isnothing∘last, Dict(pairs((;boundary,normalize,smooth_scale_2D))))
    plot(kde(samples; kde_kwargs...); kwargs...)
end