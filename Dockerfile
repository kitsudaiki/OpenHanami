FROM ubuntu:22.04@sha256:0eb0f877e1c869a300c442c41120e778db7161419244ee5cbc6fa5f134e74736

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y openssl libuuid1 libcrypto++8 libsqlite3-0 libprotobuf23 libboost1.74 && \
    apt-get clean autoclean &&\
    apt-get autoremove --yes

# hanami
COPY ./builds/binaries/hanami /usr/bin/hanami
CMD hanami
