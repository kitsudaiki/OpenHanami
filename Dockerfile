FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y openssl libuuid1 libcrypto++8 libsqlite3-0 libprotobuf23 libboost1.74 && \
    apt-get clean autoclean &&\
    apt-get autoremove --yes

# Hanami
COPY ./builds/binaries/Hanami /usr/bin/Hanami
CMD Hanami
