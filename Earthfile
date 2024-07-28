VERSION 0.8
FROM ubuntu:22.04

# configure apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

WORKDIR /code

local-setup:
    LOCALLY
    RUN git config --local core.hooksPath .githooks


image:
    BUILD +compile-code
    BUILD +build-image

docs:
    BUILD +compile-code
    BUILD +generate-code-docu
    BUILD +build-docs


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
    COPY testing testing
    COPY .flake8 .flake8
    RUN rm -rf src/sdk/python/hanami_sdk/hanami_sdk/hanami_messages/proto3_pb2.py src/sdk/python/hanami_sdk/hanami_env src/sdk/python/hanami_sdk/build
    RUN flake8 testing/python_sdk_api/sdk_api_test.py
    RUN flake8 src/sdk/python

ansible-lint:
    RUN apt-get update && \
        apt-get install -y python3 python3-pip
    RUN pip3 install ansible-lint
    COPY deploy deploy
    COPY .ansible-lint .ansible-lint
    RUN ansible-lint deploy/ansible/hanami

secret-scan:
    RUN apt-get update && \
        apt-get install -y python3 python3-pip git
    RUN pip3 install detect-secrets
    COPY . .
    RUN git ls-files -z | xargs -0 detect-secrets-hook --baseline .secrets.baseline


prepare-build-dependencies:
    RUN apt-get update && \
        apt-get install -y clang-15 \
                           make \
                           cmake \
                           bison \
                           flex \
                           git \
                           ssh \
                           libssl-dev \
                           libcrypto++-dev \
                           libboost-dev \
                           nlohmann-json3-dev \
                           uuid-dev  \
                           libsqlite3-dev \
                           protobuf-compiler \
                           # https://github.com/kitsudaiki/Hanami/issues/377
                           # nvidia-cuda-toolkit \
                           nano && \
        ln -s /usr/bin/clang++-15 /usr/bin/clang++ && \
        ln -s /usr/bin/clang-15 /usr/bin/clang
    COPY . .


compile-code:
    FROM +prepare-build-dependencies
    RUN cmake -DCMAKE_BUILD_TYPE=Release -Drun_tests=ON  .
    RUN make -j8
    RUN mkdir /tmp/Hanami && \
        find src -type f -executable -exec cp {} /tmp/Hanami \;
    SAVE ARTIFACT /tmp/Hanami /tmp/Hanami


compile-cli:
    RUN apt-get update && \
        apt-get install -y wget protobuf-compiler golang-goprotobuf-dev && \
        wget -c https://go.dev/dl/go1.22.5.linux-amd64.tar.gz && \
        tar -C /usr/local/ -xzf go1.22.5.linux-amd64.tar.gz
    COPY src src
    RUN cd ./src/sdk/go/hanami_sdk && \
        protoc --go_out=. --proto_path ../../../libraries/hanami_messages/protobuffers hanami_messages.proto3
    RUN cd src/cli/hanamictl && \
        /usr/local/go/bin/go build .
    SAVE ARTIFACT ./src/cli/hanamictl/hanamictl /tmp/hanamictl


build-image:
    ARG image_name

    RUN apt-get update && \
        apt-get install -y openssl libuuid1 libcrypto++8 libsqlite3-0 libprotobuf23 libboost1.74 libcudart11.0 && \
        apt-get clean autoclean && \
        apt-get autoremove --yes

    COPY ./builds/binaries/Hanami  /usr/bin/Hanami

    # run Hanami
    ENTRYPOINT ["/usr/bin/Hanami"]

    SAVE IMAGE "$image_name"


generate-code-docu:
    RUN apt-get update && \
        apt-get install -y openssl libuuid1 libcrypto++8 libsqlite3-0 libprotobuf23 libboost1.74 libcudart11.0
    COPY ./builds/binaries/Hanami /usr/bin/Hanami
    RUN /usr/bin/Hanami --generate_docu
    SAVE ARTIFACT open_api_docu.json /tmp/Hanami_docs/open_api_docu.json
    SAVE ARTIFACT config.md /tmp/Hanami_docs/config.md
    SAVE ARTIFACT db.md /tmp/Hanami_docs/db.md


generate-docs:
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
    COPY ./builds/docs/db.md docs/backend/db.md
    COPY ./builds/docs/config.md docs/backend/config.md
    COPY ./builds/docs/open_api_docu.json docs/frontend/open_api_docu.json

    RUN mkdocs build --clean

    SAVE ARTIFACT site AS LOCAL site


build-docs:
    ARG image_name

    RUN apt-get update && \
        apt-get install -y python3

    COPY +generate-docs/site /hanami_docs

    WORKDIR /hanami_docs

    CMD python3 -m http.server 8000

    SAVE IMAGE "$image_name"
