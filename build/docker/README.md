# Dockerbuilder

## Usage

1. build docker-image for the buil

```
docker build  -t kitsudaiki/hanami_builder:0.4.0 .
docker run -it -u root -v /var/run/docker.sock:/var/run/docker.sock kitsudaiki/hanami_builder:0.4.0 bash
./build_hanami.sh
exit
./push_images.sh
```
