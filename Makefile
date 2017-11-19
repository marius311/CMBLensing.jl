build:
	docker build -t marius311/cmblensing.jl .

start:
	docker run --rm -itp 8888 --name cmblensing -v `pwd`:/home/bayes/.julia/v0.6/CMBLensing marius311/cmblensing.jl | (sleep 1 && sed -e "s/localhost/$$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' cmblensing)/g")

stop:
	docker rm -f cmblensing
