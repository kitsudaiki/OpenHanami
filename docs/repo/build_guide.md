# How to build

## Preparation

-   Install packages

    -   For Ubuntu 22.04 and 24.04 (Debian should work too):

        ```bash
        apt-get install clang-15 make cmake bison flex libssl-dev libcrypto++-dev libboost-dev nlohmann-json3-dev uuid-dev libsqlite3-dev protobuf-compiler
        ```

    -   On Ubuntu 24.04 a new clang version than `15` can also be used or g++. See
        [supported versions](/#supported-environment)

-   Clone repository with submodules

    ```bash
    git clone --recurse-submodules https://github.com/kitsudaiki/OpenHanami.git
    ```

-   In case the repo was cloned without submodules initially:

    ```bash
    cd OpenHanami

    git submodule init
    git submodule update --recursive
    ```

## Build hamami plain

-   Compile hanami

    ```bash
    cd OpenHanami
    mkdir build
    cd build

    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8
    ```

-   In case you also want to compile the unit-tests and so on, you have to add `-Drun_tests=ON` to
    the cmake-commands

-   Resulting binary

    ```bash
    ./src/hanami/hanami
    ```

## Build hanami as docker-image

### With docker-build

Run `docker build -t <DOCKER_IMAGE_NAME> .`

!!! example

    ```bash
    docker build -t hanami:test .
    ```

### With earthly

-   Install [earthly](https://github.com/earthly/earthly)

-   The code can be build as image like this:

    ```bash
    earthly +image --image_name=<DOCKER_IMAGE_NAME>
    ```

    !!! example

        ```bash
        earthly +image --image_name=hanami:test
        ```

## Build CLI-client

-   Install [earthly](https://github.com/earthly/earthly)

- build protobuf-messages within the hanami_sdk directory

    ```bash
    earthly --artifact +compile-cli/tmp/hanamictl ./builds/
    ```

-  then you have a new local directory `builds`, where the resulting binary of the build-process is placed into

## Build python-SDK as package

- install packages

    ```bash
    sudo apt-get update
    sudo apt-get install -y protobuf-compiler python3 python3-pip
    sudo pip3 install wheel
    ```

- build protobuf-messages and package

    ```bash
    cd ./src/sdk/python/hanami_sdk/hanami_sdk
    protoc --python_out=. --proto_path ../../../../libraries/hanami_messages/protobuffers  hanami_messages.proto3
    cd ..
    python3 setup.py bdist_wheel --universal
    ```

## Prechecks

There are a bunch of pre-checks at the beginning of the CI-pipeline, which can fail and where it is useful to be able to run the same tests locally for debugging. Nearly all of them use [earthly](https://github.com/earthly/earthly)

### Flake8-check

- run `earthly --ci +flake8`

### Secret-scan

- run `earthly --ci +secret-scan`

It is possible, that the check fails, even if there are no (new) secrets in the code and fails, because of some other code-movements. The check compares all to the `.secrets.baseline`-file, where also line-numbers are marked. To update the file to get the test green again:

- install [detect-secrets](https://github.com/Yelp/detect-secrets)

- update file with `detect-secrets scan > .secrets.baseline`

### Ansible-lint

- run `earthly --ci +ansible-lint`

### Cpp-check

- run `earthly --ci +cppcheck`

### Clang-format check

- run `earthly --ci +clang-format`

## Build docs

-   The documenation can be build as image like this:

    ```bash
    earthly +docs --image_name=<DOCKER_IMAGE_NAME>
    ```

    !!! example

        ```bash
        earthly +docs --image_name=hanami_docs:test
        ```

-   The documentation listen on port 8000 within the docker-container. So the port has to be
    forwarded into the container:

    ```bash
    docker run -p 127.0.0.1:8080:8000 hanami_docs:test
    ```

-   After this within the browser the addess `127.0.0.1:8080` can be entered to call the
    documenation within the browser.

## Run preview of docs

-   Install Mkdocs and plugins

    ```bash
    pip3 install mkdocs mkdocs-material mkdocs-swagger-ui-tag mkdocs-drawio-exporter
    ```

-   To build the documentation `Draw.io` also has to be installed on the system

    -   Example how to install draw.io

        ```bash
        curl -s https://api.github.com/repos/jgraph/drawio-desktop/releases/latest | grep browser_download_url | grep "amd64"  | grep "deb" | cut -d "\"" -f 4 | wget -i -

        apt -f -y install ./drawio-amd64-*.deb
        ```

-   checkout repository and run the local server

    ```bash
    git clone --recurse-submodules https://github.com/kitsudaiki/OpenHanami.git
    cd OpenHanami

    mkdocs serve
    ```

-   Open web-browser with address `http://127.0.0.1:8000/` to see the docs. The
    `mkdocs serve`-command runs in the background and makes live-updates of all changes within the
    files.
