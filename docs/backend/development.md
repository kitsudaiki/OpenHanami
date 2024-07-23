# Development

This document should help to setup a local environment for development.

!!! note

    If there are still things not clear or don't work for you with this documentation, or when you find any bug or problem, then please create an issue on github.

## Prepare local setup

- Clone repository:

    ```
    git clone https://github.com/kitsudaiki/Hanami.git
    ```

- Install dependencies

    ```
    apt-get install gcc g++ clang-15 clang-format-15 make bison flex libssl-dev libcrypto++-dev libboost1.74-dev uuid-dev libsqlite3-dev protobuf-compiler
    ```

- Install `cmake`

- Download the MNIST-dataset

    For testing the standard MNIST-dataset is used. Download the 4 files somewhere from the web:

    ```
    train-images.idx3-ubyte
    train-labels.idx1-ubyte
    t10k-images.idx3-ubyte
    t10k-labels.idx1-ubyte
    ```

    !!! info 

        I had downloaded the files in the past from `https://yann.lecun.com/exdb/mnist/`, but for an unknown reason this site now requires a login. I'm not sure, if I'm allowed to host this files by myself and provide them here, or if this would violate copyrights. So you have to look for yourself to find a source for these files.

- Copy basic config-files

    ```
    cp example_configs/hanami /etc/

    sudo chown -R $(id -u):$(id -g) /etc/hanami
    ```

- Update config-files

    1. In file `/etc/hanami/hanami.conf` update the entry `dashboard_files` to the absolute path to the `Hanami-Dashboard/src` within the Hanami reposotory, which you checked out in the first step

    2. In file `/etc/hanami/hanami_testing.conf` update the path to the 4 MNIST files, so they point to the MNIST-files, which you downloaded eralier.

## Build project

### Prepare

For the initial run, environment-variables have to be set to initialize the sqlite-database and create the first admin-user:

```
export HANAMI_ADMIN_USER_ID=asdf
export HANAMI_ADMIN_USER_NAME=asdf
export HANAMI_ADMIN_PASSWORD=asdfasdf
```

(These are the default testing-configs, to match with the config-files, which are copied above.)

### With Qt Creator

When using the Qt Creator as IDE, then you only need to load the `Hanami.pro` in the root directory of the repository and select `clang-15` and `clang++-15` as compiler. The unit-, function- and memory-leak-tests of the libraries are not compiled by default. Within the project-settings you have to add as additional build argument the linke `CONFIG+=run_tests` to build the tests too.

### Without Qt Creator

Without the Qt creator you can still build the project within the root-directory of the repository with:

```
cmake -DCMAKE_BUILD_TYPE=Debug -Drun_tests=ON "Hanami/"

make -j8
```

## Testing

There is a python-script, which uses the python-version fo the SDK to run basic tests against the API.

- install python:

```
sudo apt-get install python3 python3-pip python3-venv
```

- go into the test-directory 

`cd Hanami/testing/python_sdk_api`


- create and prepare python-env

```
python3 -m venv hanami_env

source hanami_env/bin/activate

pip3 install -r hanami_sdk/requirements.txt 
```

- run tests

`./sdk_api_test.py`

