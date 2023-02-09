from ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get update && \
    apt-get install -y \
        gcc \
        g++ \
        make \
        qt5-qmake \
        bison \
        flex \
        xxd \
        opencl-headers \
        ocl-icd-opencl-dev \
        libssl-dev \
        libcrypto++-dev \
        libboost1.74-dev \
        uuid-dev \
        libsqlite3-dev \
        protobuf-compiler \ 
        nano \
        git \
        ssh
