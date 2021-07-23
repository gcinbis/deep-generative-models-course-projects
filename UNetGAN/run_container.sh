#!/bin/bash

XSOCK=/tmp/.X11-unix
XAUTH=$HOME/.Xauthority

PROJECT_NAME="testcahit"

echo "Update docker image"
nvidia-docker build -f docker/Dockerfile -t $PROJECT_NAME .

IMAGE_NAME="$PROJECT_NAME:latest"

WORKDIR=$(pwd)
VOLUMES="--volume=$XSOCK:$XSOCK:rw
         --volume=$XAUTH:$XAUTH:rw
         --volume=$WORKDIR:/workspace/$PROJECT_NAME"
DEVICES="--device=/dev/nvidia0:/dev/nvidia0
         --device=/dev/nvidiactl:/dev/nvidiactl
         --device=/dev/nvidia-uvm:/dev/nvidia-uvm"


USER_ID=1000
RUNTIME="--runtime=nvidia"
#DISPLAY=127.0.0.1:0

## Allow for x-forwarding
## Suppresses QXcbConnection: XCB error: 2 (BadValue) errors
## Probably unsafe
xhost +local:docker

echo "Loading image $IMAGE_NAME"

nvidia-docker run \
    -dt --rm \
    $DEVICES \
    $VOLUMES \
    --env="XAUTHORITY=${XAUTH}" \
    --env="DISPLAY=${DISPLAY}" \
    --env="USER_ID=$USER_ID" \
    --network="host" \
    --privileged \
    $RUNTIME \
    --entrypoint /bin/bash \
    $IMAGE_NAME
