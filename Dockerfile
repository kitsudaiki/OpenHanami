FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y openssl libuuid1 libcrypto++8 libsqlite3-0 libprotobuf23 libboost1.74 libcudart11.0 && \
    apt-get clean autoclean &&\
    apt-get autoremove --yes

COPY src/frontend/Hanami-Dashboard /etc/Hanami-Dashboard

# Remove symlinks in order to replace them by the real repos
RUN rm /etc/Hanami-Dashboard/src/sdk /etc/Hanami-Dashboard/src/hanami_messages /etc/Hanami-Dashboard/src/Hanami-Dashboard-Dependencies

# Dashboard
COPY src/sdk /etc/Hanami-Dashboard/src/sdk
COPY src/libraries/hanami_messages /etc/Hanami-Dashboard/src/hanami_messages
COPY src/frontend/Hanami-Dashboard-Dependencies /etc/Hanami-Dashboard/src/Hanami-Dashboard-Dependencies

# Hanami
COPY src/Hanami/Hanami /usr/bin/Hanami
CMD Hanami
