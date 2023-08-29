SAHI=/mnt/c/Users/CJIAHA1/dev/yh-yolov7/yolov7/sahi

docker run -it --rm \
	--gpus all \
	-v $SAHI:/sahi \
	yolov7
