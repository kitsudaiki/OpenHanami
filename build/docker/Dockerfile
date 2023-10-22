FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y clang-15 \
                       gcc \
                       g++ \
                       make \
                       qt5-qmake \
                       bison \
                       flex \
                       git \
                       ssh \
                       libssl-dev \
                       libcrypto++-dev \
                       libboost1.74-dev \
                       uuid-dev  \
                       libsqlite3-dev \
                       protobuf-compiler \
                       nvidia-cuda-toolkit \
                       nano \
    && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common && \ 
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update  && \
    apt-get install -y docker-ce \
    && \
    apt-get clean autoclean &&\
    apt-get autoremove --yes

RUN ln -s /usr/bin/clang++-15 /usr/bin/clang++ && \
    ln -s /usr/bin/clang-15 /usr/bin/clang
RUN mkdir /dockerbuilder
COPY files/build_hanami.sh /dockerbuilder 
COPY files/Dockerfile_hanami /dockerbuilder
COPY files/Dockerfile_hanami_base /dockerbuilder

WORKDIR /dockerbuilder
