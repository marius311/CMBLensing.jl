#!/bin/sh

# this is a way to get the name of the image just built by the docker/build-push-action 
# presumably there will be a better way if/once this is addressed:
# https://github.com/docker/build-push-action/issues/24
IMAGE_NAME=$(docker images marius311/cmblensing.jl --format "{{.Repository}}:{{.Tag}}" | head -n 1)

docker run \
    -e GITHUB_TOKEN=$GITHUB_TOKEN \
    -e GITHUB_REPOSITORY=$GITHUB_REPOSITORY \
    -e GITHUB_EVENT_NAME=$GITHUB_EVENT_NAME \
    -e GITHUB_REF=$GITHUB_REF \
    -e GITHUB_ACTOR=$GITHUB_ACTOR \
    -e IMAGE_NAME=$IMAGE_NAME \
    -v $(pwd)/.git:/home/cosmo/CMBLensing/.git \
    -w /home/cosmo/CMBLensing/docs \
    $IMAGE_NAME \
    julia make.jl
