LOCAL_DIR=/home/user/dev/aipipeline/cv-usecase/pipeline-manager/data

docker run -d --rm \
  --shm-size=2g \
  --ipc=shareable \
  --name=pipeline-manager \
  -v $LOCAL_DIR:/data \
	cv-pipeline-manager:v1

docker run -it --rm \
	--gpus all \
  --ipc=container:pipeline-manager \
	detector-component:v1
