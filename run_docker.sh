docker run -d --rm \
  --shm-size=2g \
  --ipc=shareable \
  --name=pipeline-manager \
	cv-pipeline-manager:v1

docker run -it --rm \
	--gpus all \
  --ipc=container:pipeline-manager \
	detector-component:v1
