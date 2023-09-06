IMAGE_DIR=/home/user/dev/aipipeline/cv-usecase/pipeline-manager/images

docker run -d --rm \
  --shm-size=2g \
  --ipc=shareable \
  --name=pipeline-manager \
  -v $IMAGE_DIR:/images \
	cv-pipeline-manager:v1

docker run -it --rm \
	--gpus all \
  --ipc=container:pipeline-manager \
	detector-component:v1
