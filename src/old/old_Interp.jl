module Interp
using PyCall
export smooth, meshgrid, spline_interp2, nearest_interp2, linear_interp1

@pyimport scipy.interpolate as scii

smooth(x, y) = scii.UnivariateSpline(x,y)[:__call__](x)


function meshgrid(side_x,side_y)
	x = repmat(reshape([side_x],(1,length(side_x))) ,length(side_y),1)
	y = repmat(reshape([side_y],(length(side_y),1)) ,1,length(side_x))
	x,y
end

function  perodic_padd_xy(x,y,pad_proportion = 0.05)
	maxx = x[1,end];
	minx = x[1,1];
	maxy = y[end,1];
	miny = y[1,1];
	Deltx = x[2,2]-x[1,1];
	Delty = y[2,1]-y[1,1];

	npad = round(pad_proportion*size(x,1));
	middle_x_section =    [x[:,(end-npad+1):end] .- maxx .+ minx .- Deltx x x[:,1:npad] .- minx .+ maxx .+ Deltx];
	full_x = [middle_x_section[1:npad,:]; middle_x_section; middle_x_section[1:npad,:] ];
	
	middle_y_section = [ y[(end-npad+1):end,:] .- maxy .+ miny .- Delty ; y ; y[1:npad,:] .- miny .+ maxy .+ Delty ];
	full_y =  [middle_y_section[:,1:npad] middle_y_section middle_y_section[:,1:npad] ];
	
	full_x, full_y
end


function  perodic_padd_z(z,pad_proportion = 0.05)
	npad = int(pad_proportion*size(z,1));
	z_right = z[:, (end-npad+1):end]
	z_left = z[:, 1:npad];
	z_top = z[1:npad,:];
	z_bottom = z[(end-npad+1):end,:];
	z_top_left = z[1:npad, 1:npad];
	z_bottom_left = z[(end-npad+1):end,1:npad];
	z_top_right = z[1:npad, (end-npad+1):end];
	z_bottom_right = z[(end-npad+1):end, (end-npad+1):end];
	full_z = [z_bottom_right z_bottom z_bottom_left;z_right z z_left;z_top_right z_top z_top_left];
	full_z
end
function  perodic_padd(x,y,z,pad_proportion = 0.05)
	full_x,full_y = perodic_padd_xy(x,y,pad_proportion)
	full_z = perodic_padd_z(z,pad_proportion)
	full_x,full_y,full_z 
end



function  spline_interp2(xtmp, ytmp, ztmp, xi, yi, pad_proportion = 0.02)
	xx, yy, zz = perodic_padd(xtmp, ytmp, ztmp, pad_proportion)
	iz = scii.RectBivariateSpline(xx[1,:], yy[:,1], zz, s=0)[:ev]
	zi = convert(Vector{Float64},iz(yi[:], xi[:]))
	return reshape(zi, size(xi))
end



function  nearest_interp2(xtmp,ytmp,ztmp,xi,yi; pad_proportion = 0.05)
	x,y,z = perodic_padd(xtmp,ytmp,ztmp,pad_proportion)
	xmin = x[1,1];
	ymin = x[1,1];
	xdelt = x[2,2]-x[1,1];
	ydelt = y[2,1]-y[1,1];

	zi = Array(Float64,size(xi));
	ind_x =  int(round((xi .- xmin)./xdelt .+ 1));
	ind_y =  int(round((yi .- ymin)./ydelt .+ 1));
	for j=1:size(xi,2)
		for i=1:size(xi,1)
			zi[i,j] = z[ind_y[i,j], ind_x[i,j]];
		end
	end
	return zi
end

 function linear_interp1(x,z,xi)
	# assumes that x is a monotonic grid
	xitmp = xi[:]
	deltx = x[2] - x[1]
	xi_upindex = int( ceil(xitmp/deltx) .- x[1]/deltx .+ 1 )
	xi_downindex = xi_upindex .- 1
	idown = find(xi_downindex .< 1)
	xi_upindex[idown] = 2
	xi_downindex[idown] = 1
	iup = find(xi_upindex .> endof(x))
	xi_upindex[iup]=endof(x)
	xi_downindex[iup]=endof(x)-1
	
	slope=(z[xi_upindex] - z[xi_downindex]) ./ (x[xi_upindex] - x[xi_downindex])
	answer = slope .* (xitmp .- x[xi_downindex]) .+ z[xi_downindex]
	reshape(answer,size(xi))
end

end

