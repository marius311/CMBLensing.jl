# This needs to be set for double-struck operators in the PGFPlotsX
# backend. How to only run if PGFPlotsX.jl is installed, and perhaps 
# also only if the that backend is set.
#push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")

function tick_locations(Npix, θpix)
    # calculate tick locations in pixels and in degrees
    # used to set axis guides
    θlim(Npix) = div(floor(Int, Npix * θpix / 60), 2)
    θ₊ = θlim(Npix)
    δθ = θlim(Npix) < 10 ? 1 : 5
    θticks = δθ:δθ:θ₊
    θticks = [-reverse(θticks); 0;  θticks]
    pixticks = (θticks .+ (θpix * Npix / 60 / 2)) .* (60 / θpix)
    latex_θticks = [L"%$i^\circ" for i in θticks]
    pixticks, latex_θticks
end

@recipe function _(m::FlatS0; data_scale=1.)
    # initial heatmap plotting
    seriestype :=  :heatmap
    seriescolor --> :vik
    aspect_ratio := :equal
   
    # layout of subplots 
    size --> (500, 400)

    @unpack θpix, Nx, Ny = fieldinfo(m)

    # sort out x and y ticks
    xlims --> (1, Nx) # this is because the default settings add padding in the x direction
    ylims --> (1, Ny) # there is probably a better way of turning that off

    # apply tick labels
    xticks := tick_locations(Nx, θpix)
    yticks := tick_locations(Ny, θpix)
    tick_direction := :out

    # choices about axes
    axis --> true
    framestyle --> :box 
    grid := true

    # labeling
    xguide --> L"x"
    yguide --> L"y"

    # colorbar
    colorbar_title --> L"\mu \mathrm{K~amin}"

    arr = Array(m[:Ix]) .* data_scale
    #clim = floor(Int, quantile(abs.(arr[@. !isnan(arr)][:]),0.999))
    #clims := (-clim, clim)
    #title := L"Map~(x, y)~[%$Nx \times %$Ny~@~%$θpix\prime]"
    arr
end

@recipe function _(m::FlatS2)
    # initial heatmap plotting
    seriestype :=  :heatmap
    seriescolor --> :vik
    aspect_ratio := :equal
   
    # layout of subplots 
    size --> (1000, 400)
    layout := (1, 2)

    @unpack θpix, Nx, Ny = fieldinfo(m)

    # sort out x and y ticks
    xlims --> (1, Nx) # this is because the default settings add padding in the x direction
    ylims --> (1, Ny) # there is probably a better way of turning that off

    xticks := tick_locations(Nx, θpix)
    yticks := tick_locations(Ny, θpix)
    tick_direction := :out

    # choices about axes
    axis --> true
    axisstyle --> :box 

    # labeling
    xguide --> L"x"
    yguide --> L"y"

    # colorbar
    #colorbar_title --> L"\mu \mathrm{K~amin}"

    @series begin 
        subplot := 1
        arr = Array(m[:Ex])
        clim = floor(Int, quantile(abs.(arr[@. !isnan(arr)][:]),0.999))
        clims := (-clim, clim)
        #title := L"E~(x, y)~[%$Nx \times %$Ny~@~%$θpix\prime]"
        arr
    end

    @series begin
        subplot := 2
        arr = Array(m[:Bx])
        clim = floor(Int, quantile(abs.(arr[@. !isnan(arr)][:]),0.999))
        clims := (-clim, clim)
        #title := L"B~(x, y)~[%$Nx \times %$Ny~@~%$θpix\prime]"
        arr
    end
end

@recipe function _(Cℓ::InterpolatedCℓs, ℓ)
    # labeling
    xguide --> L"\ell"
    yguide --> L"\frac{\ell(\ell + 1)}{2\pi} C_\ell~[\mathrm{\mu ~K}^2]"

    xscale --> :log10
    yscale --> :log10

    ℓ, Cℓ(ℓ)
end