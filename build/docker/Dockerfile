FROM ubuntu:22.04

RUN apt-get update && apt-get install -y gcc g++ make qt5-qmake bison flex xxd git ssh libssl-dev libcrypto++-dev libboost1.74-dev uuid-dev  libsqlite3-dev protobuf-compiler protobuf-compiler nano ocl-icd-opencl-dev opencl-headers nvidia-cuda-toolkit
RUN apt-get install -y apt-transport-https ca-certificates curl software-properties-common && \ 
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update  && \
    apt-get install -y docker-ce

RUN mkdir /dockerbuilder
COPY files/build_components.sh /dockerbuilder 
COPY files/Dockerfile_base /dockerbuilder
COPY files/Dockerfile_hanami /dockerbuilder
COPY files/hanami_run.sh /dockerbuilder
COPY files/Dockerfile_dashboard /dockerbuilder

WORKDIR /dockerbuilder
