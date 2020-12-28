@recipe function _(m::FlatS2)
    # initial heatmap plotting
    seriestype   :=  :heatmap
    seriescolor  --> :vik
    aspect_ratio := :equal
   
    # layout of subplots 
    size --> (1000, 400)
    layout := (1, 2)

    @unpack θpix, Nx, Ny = fieldinfo(m)

    # sort out x and y ticks
    xlims --> (1, Nx) # this is because the default settings add padding in the x direction
    ylims --> (1, Ny) # there is probably a better way of turning that off

    θlim(Npix) = div(floor(Int, Npix * θpix / 60), 2) # get extent for x / y ticks
    deg_ticks(Npix) =  [L"%$i^\circ" for i in -θlim(Npix):θlim(Npix)] # make deg tick labels
    pix_tick(Npix) = range(1, Npix, length= 2 * θlim(Npix) + 1)  # get positions of new ticks in pixels

    xticks :=  (pix_tick(Nx), deg_ticks(Nx))
    yticks :=  (pix_tick(Ny), deg_ticks(Ny))

    # choices about axes
    axis --> true
    framestyle --> :box 

    # labeling
    xguide --> L"x"
    yguide --> L"y"

    # colorbar
    colorbar_title --> L"\mu \mathrm{K~amin}"

    @series begin 
        subplot := 1
        arr = Array(m[:Ex])
        clim = floor(Int, quantile(abs.(arr[@. !isnan(arr)][:]),0.999))
        clims := (-clim, clim)
        title := L"E~(x, y)~[%$Nx \times %$Ny~@~%$θpix\prime]"
        arr
    end

    @series begin
        subplot := 2
        arr = Array(m[:Bx])
        clim = floor(Int, quantile(abs.(arr[@. !isnan(arr)][:]),0.999))
        clims := (-clim, clim)
        title := L"B~(x, y)~[%$Nx \times %$Ny~@~%$θpix\prime]"
        arr
    end
end