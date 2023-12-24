# Hanami

![Github workfloat status](https://img.shields.io/github/actions/workflow/status/kitsudaiki/Hanami/build_test.yml?branch=develop&style=flat-square&label=build%20and%20test)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/kitsudaiki/Hanami?label=version&style=flat-square)
![GitHub](https://img.shields.io/github/license/kitsudaiki/Hanami-AI?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Linux--x64-lightgrey?style=flat-square)

<p align="center">
  <img src="assets/hanami-logo-with-text.png" width="500" height="594" />
</p>

# **IMPORTANT: This project is still an experimental prototype at the moment and NOT ready for productive usage.** 

There is still a huge bunch of known bugs and missing validations, which can break the backend. Even the documentation here is quite basic. Normally I absolutely dislike it to make something public, which has known bugs and other problems, but I simply don't wanted to wait longer for the open-sourcing of this project. Keep in mind, that this project is created by a single person in his spare time beside a 40h/week job. ;)

## Intro

Hanami is an AI-as-a-Service project, based on a concept created by myself. It is written from scratch with a Backend in C++ with Web-frontend.

The actual prototype consists of:

- partially implementation of an own concept for an artificial neuronal network. It has no fixed connections between the nodes, but creates connections over time while learning. Additionally it doesn't need a normalization of input-values and this way it can also handle unknown data as input. This should make it flexible and efficient. The current state is extremely experimental.
- very experimental but working GPU-support with CUDA ~~and OpenCL~~
- multi-user- and multi-project-support, so multiple-users can share the same physical host
- able to generate an OpenAPI-documentation from source-code
- basic energy-optimization supporting the scheduling of threads of all components and changing the cpu-frequency based on workload
- basic monitoring of cpu-load
- sdk-library written in python
- basic cli-client written in go
- Webfrontend with client-side rendering and SDK-library
- Websocket-connection to directly interact with the artificial neuronal networks
- CI-pipelines, Test-Tool, Docker-build-process and basic helm-chart to deploy the project on Kubernetes

## First benchmark

Test-case:

- Dataset: MNIST handwritten letters
- Hardware: Intel i7-1165G7 and 16GB RAM with 3200MT/s
- Settings: 
    - **CPU** with **one processing thread** 
    - **no batches**, so each of image is processed one after the other
    - values are pushed directly into the network without normalization between 0 and 1
    - average of 10 measurements


|             |      average result        |
| ----------- | ------------------------------------ |
| time for train-dataset (60000 Images); 1. epoch  | 1.9 s |
| time for test-dataset (10000 Images)       |  0.1 s |
| accuracy of test-dataset after 1. epoch   |  94.21 % |
| accuracy of test-dataset after 10. epoch   |  96.43 % |

## Possible use-case

Because the normalization of input is not necessary, together with the good performance of training single inputs (based on the benchmark) and the direct interaction remotely over websockets, could make this project useful for processing measurement-data of sensors of different machines, especially for new sensors, where the exact maximum output-values are unknown. So continuous training of the network right from the beginning would be possible, without collecting a bunch of data at first.

## Documentation

All of this page and more in the documentation on: 

https://docs.hanami-ai.com

- Installation-Guide to deploy HanamiAI on a kubernetes for testing:

    [Installation on Kubernetes](https://docs.hanami-ai.com/how_to/installation/)

- Automatic generated OpenAPI documentation

    [OpenAPI docu](https://docs.hanami-ai.com/api/rest_api_documentation/)

- To get a first impression there is a first example-workflow via the dashboard:

    [Dashboard](https://docs.hanami-ai.com/how_to/dashboard/)

- Many basic dependencies were created in context of this project. Here is an overview of all involved repositories:

    [Dependency-Overview](https://docs.hanami-ai.com/other/dependencies/)

If you need help to setup things, have a question or something like this, feel free to contact me by eMail or use the `Question`-template in the issues.

## Issue-Overview

[Hanami-Project](https://github.com/users/kitsudaiki/projects/9/views/4)

## This repository

**Required packages:**

```
sudo apt-get install -y git ssh gcc g++ clang-15 clang-format-15 make cmake bison flex libssl-dev libcrypto++-dev libboost1.74-dev uuid-dev  libsqlite3-dev protobuf-compiler nvidia-cuda-toolkit
```

**Clone repo with:**

```
git clone --recurse-submodules https://github.com/kitsudaiki/Hanami.git
cd Hanami

# load pre-commit hook
git config core.hooksPath .git-hooks
```

**In case the repo was cloned without submodules initially:**

```
git submodule init
git submodule update --recursive
```

**Mkdocs and plugins:**

```
pip3 install mkdocs-material mkdocs-swagger-ui-tag mkdocs-drawio-exporter
```

(to build the documentation `Draw.io` also has to be installed on the system)

## Author

**Tobias Anker**

eMail: tobias.anker@kitsunemimi.moe

## License

The complete project is under [Apache 2 license](https://github.com/kitsudaiki/Hanami/blob/develop/LICENSE).

## Contributing

If you want to help the project by contributing things, you can check the [Contributing guide](https://github.com/kitsudaiki/Hanami/blob/develop/CONTRIBUTING.md).
