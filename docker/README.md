#To build the docker image with kunai ready make sure you have docker.io and docker-compose installed


then enter the directory KUNAI-static-analyzer/docker and type

docker-compose build
docker-compose run

and you will get an image with the project built with LLVM17.0.1 and MLIR
