VERSION 0.8
FROM ubuntu:22.04

# configure apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

WORKDIR /code

#=================================================================================
#                               INTERNAL
#=================================================================================

local-setup:
    LOCALLY
    RUN git config --local core.hooksPath .githooks

prepare-build-dependencies:
    RUN apt-get update && \
        apt-get install -y clang-15 \
                           gcc \
                           g++ \
                           make \
                           cmake \
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
                           nano && \
        ln -s /usr/bin/clang++-15 /usr/bin/clang++ && \
        ln -s /usr/bin/clang-15 /usr/bin/clang
    COPY . .

compile:
    FROM +prepare-build-dependencies
    RUN cmake -DCMAKE_BUILD_TYPE=Release .
    RUN make -j8
    SAVE ARTIFACT src/Hanami/Hanami AS LOCAL Hanami

compile-with-tests:
    FROM +prepare-build-dependencies
    RUN cmake -DCMAKE_BUILD_TYPE=Release -Drun_tests=ON  .
    RUN make -j8
    SAVE ARTIFACT src/Hanami/Hanami AS LOCAL Hanami

generate-code-docu:
    FROM +compile
    COPY +compile/Hanami /usr/bin/Hanami
    RUN /usr/bin/Hanami --generate_docu
    SAVE ARTIFACT open_api_docu.json AS LOCAL open_api_docu.json
    SAVE ARTIFACT config.md AS LOCAL config.md
    SAVE ARTIFACT db.md AS LOCAL db.md

build-docs:
    RUN apt-get update && \
        apt-get install -y python3 \
                           python3-pip \
                           wget \
                           curl && \
        pip3 install mkdocs \
                     mkdocs-material \
                     mkdocs-swagger-ui-tag \
                     mkdocs-drawio-exporter && \
        curl -s https://api.github.com/repos/jgraph/drawio-desktop/releases/latest | grep browser_download_url | grep "amd64"  | grep "deb" | cut -d "\"" -f 4 | wget -i - && \ 
        apt -f -y install ./drawio-amd64-*.deb

    COPY mkdocs.yml .
    COPY CHANGELOG.md .
    COPY LICENSE .
    COPY docs docs
    COPY +generate-code-docu/db.md docs/backend/db.md
    COPY +generate-code-docu/config.md docs/backend/config.md
    COPY +generate-code-docu/open_api_docu.json docs/frontend/open_api_docu.json

    RUN mkdocs build --clean

    SAVE ARTIFACT site AS LOCAL site


#=================================================================================
#                               PUBLIC
#=================================================================================

cppcheck:
    RUN apt-get update && \
        apt-get install -y cppcheck
    COPY src src
    RUN rm -rf \
          src/libraries/hanami_messages/protobuffers/hanami_messages.proto3.pb.h
    RUN cppcheck --error-exitcode=1 src/Hanami
    RUN cppcheck --error-exitcode=1 src/libraries

clang-format:
    RUN apt-get update && \
        apt-get install -y clang-format-15
    COPY src src
    RUN rm -rf \
          src/sdk/python \
          src/third-party-libs \
          src/libraries/hanami_messages/protobuffers/hanami_messages.proto3.pb.h
    COPY .clang-format .
    RUN find . -regex '.*\.\(h$\|c$\|hpp$\|cpp$\)' | while read f; do \
              clang-format-15 -style=file:.clang-format --dry-run --Werror $f; \
              if [ $? -ne 0 ]; then \
                  exit 1; \
              fi; done

flake8:
    RUN apt-get update && \
        apt-get install -y python3 python3-pip
    RUN pip3 install flake8
    COPY src src
    COPY .flake8 .
    RUN flake8 testing/python_sdk_api/sdk_api_test.py
    RUN flake8 src/sdk/python

ansible-lint:
    RUN apt-get update && \
        apt-get install -y python3 python3-pip
    RUN pip3 install ansible-lint
    COPY src src
    COPY .ansible-lint .
    RUN ansible-lint deploy/ansible/hanami

build:
    ARG docker_tag

    RUN apt-get update && \
        apt-get install -y openssl libuuid1 libcrypto++8 libsqlite3-0 libprotobuf23 libboost1.74 libcudart11.0 && \
        apt-get clean autoclean && \
        apt-get autoremove --yes

    COPY +compile/Hanami /usr/bin/Hanami

    COPY src/frontend/Hanami-Dashboard /etc/Hanami-Dashboard

    # Remove symlinks in order to replace them by the real repos
    RUN rm /etc/Hanami-Dashboard/src/sdk /etc/Hanami-Dashboard/src/hanami_messages /etc/Hanami-Dashboard/src/Hanami-Dashboard-Dependencies

    # copy real-data to where the symlinks were
    COPY src/sdk /etc/Hanami-Dashboard/src/sdk
    COPY src/libraries/hanami_messages /etc/Hanami-Dashboard/src/hanami_messages
    COPY src/frontend/Hanami-Dashboard-Dependencies /etc/Hanami-Dashboard/src/Hanami-Dashboard-Dependencies

    # run Hanami
    ENTRYPOINT ["/usr/bin/Hanami"]

    SAVE IMAGE --push "kitsudaiki/hanami:$docker_tag"

docs:
    ARG docker_tag

    RUN apt-get update && \
        apt-get install -y python3

    COPY +build-docs/site /hanami_docs

    WORKDIR /hanami_docs

    CMD python3 -m http.server 8000

    SAVE IMAGE --push "kitsudaiki/hanami_docs:$docker_tag"
