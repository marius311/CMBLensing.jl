


#------------------------------------ FFT functions
function fftd{T,dm}(fx::Array{T,dm}, deltx)
	c  = complex( (deltx / √(2π))^dm )
	fk = fft(fx)
	scale!(fk, c)
	fk::Array{Complex{Float64},dm}
end

function  ifftd{T,dm}(fk::Array{T,dm}, deltk)
	c = (deltk / √(2π))^dm
	fx = bfft(fk)
	scale!(fx, c)
	fx::Array{Complex{Float64},dm}
end

function  ifftdr{T,dm}(fk::Array{T,dm}, deltk)
	c = (deltk / √(2π))^dm
	fx = real(bfft(fk))
	scale!(fx, c)
	fx::Array{Float64,dm}
end




#=

FFTW.set_num_threads(CPU_CORES)
datac = im * rand(1024, 1024) 
datar = im * rand(1024, 1024) 

FFT     = plan_fft(datac; flags = FFTW.PATIENT, timelimit = 10)
IFFT    = plan_ifft(datac; flags = FFTW.PATIENT, timelimit = 10)

datac = im * rand(1024, 1024) 
datar = im * rand(1024, 1024) 

@time FFT \ datac; 
@time FFT \ datac; 
@time IFFT * datac; 
@time IFFT * datac; 
@time bfft(datac);
@time bfft(datac);

=#