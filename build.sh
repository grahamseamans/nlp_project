container_name="nlp_container"
working_dir="/home/"
image_name="nlp_image"


sudo rm -r -f .cache
sudo rm -f .bash_history
docker build -t $image_name .
docker rm --force $container_name
nvidia-docker run -d -it -v $PWD:$working_dir -w $working_dir --name $container_name --gpus all $image_name
