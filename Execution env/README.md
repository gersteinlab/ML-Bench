You can evaluate your own model by downloading the following docker 

https://ml-bench-docker.s3.amazonaws.com/ml-bench-docker/ml_bench.tar

by running

`docker load -i ml_bench.tar`

`docker run -d --name your_container_name -p host_port:container_port your_image_name`

The .tar file has a README file for instructions, and it stores the results of our experiments.
