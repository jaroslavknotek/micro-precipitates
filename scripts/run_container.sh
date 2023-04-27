sudo mount /dev/sda2 /large_disk/

docker stop tf
docker remove tf

docker run \
    -it \
    --gpus all \
	--name tf \
	--mount type=bind,source="$HOME/source",target=/source \
	--mount type=bind,source="/large_disk",target=/large_disk \
	tensorflow/tensorflow:latest-gpu /bin/bash
#	-u $(id -u):$(id -g) \
# todo run script
#    bash -c "python -m venv /tmp/.venv & cd /source/jaroslavknotek/micro-precipitates & python -m pip install -e ."
    

