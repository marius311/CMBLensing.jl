build:
	docker build -f Dockerfile.alpine -t baylens .

start:
	docker run --rm -itp 8888 --name baylens -v `pwd`:/home/bayes/.julia/CMBLensing baylens | (sleep 1 && sed -e "s/localhost/$$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' baylens)/g")

stop:
	docker rm -f baylens
