# Hanami

![Github workfloat status](https://img.shields.io/github/actions/workflow/status/kitsudaiki/Hanami/build_test.yml?branch=develop&style=flat-square&label=build%20and%20test)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/kitsudaiki/Hanami?label=version&style=flat-square)
![GitHub](https://img.shields.io/github/license/kitsudaiki/Hanami-AI?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Linux--x64-lightgrey?style=flat-square)

![Logo](assets/hanami-logo-with-text.png)

# **IMPORTANT: This project is still an experimental prototype at the moment and NOT ready for productive usage.** 

There is still a huge bunch of known bugs and missing validations, which can break the backend. Even the documentation here is quite basic. Normally I absolutely dislike it to make something public, which has known bugs and other problems, but I simply don't wanted to wait longer for the open-sourcing of this project. Keep in mind, that this project is created by a single person in his spare time beside a 40h/week job. ;)

## Intro

Hanami is an AI-as-a-Service project, based on a concept created by myself. It is written from scratch with a Backend in C++ with Web-frontend.

The actual prototype consists of:

- partially implementation of an own concept for an artificial neuronal network. It has no fixed connections between the nodes, but creates connections over time while learning. Additionally it doesn't need a normalization of input-values and this way it can also handle unknown data as input. This should make it flexible and efficient. The current state is extremely experimental.
- very experimental but working GPU-support with CUDA and OpenCL
- multi-user- and multi-project-support, so multiple-users can share the same physical host
- able to generate an OpenAPI-documentation from source-code
- basic energy-optimization supporting the scheduling of threads of all components and changing the cpu-frequency based on workload
- basic monitoring of cpu-load
- Webfrontend with client-side rendering and SDK-library
- Websocket-connection to directly interact with the artificial neuronal networks
- CI-pipelines, Test-Tool, Docker-build-process and basic helm-chart to deploy the project on Kubernetes

## Documentation

All of this page and more in the documentation on: 

https://docs.hanami-ai.com

- Installation-Guide to deploy HanamiAI on a kubernetes for testing:

    https://docs.hanami-ai.com/How_To/1_installation/

- Automatic generated OpenAPI documentation

    https://docs.hanami-ai.com/How_To/3_rest_api_documentation/

- To get a first impression there is a first example-workflow via the dashboard:

    https://docs.hanami-ai.com/How_To/2_dashboard/

- Many basic dependencies were created in context of this project. Here is an overview of all involved repositories:

    https://docs.hanami-ai.com/other/1_dependencies/



## Issue-Overview

[Hanami-Project](https://github.com/users/kitsudaiki/projects/9/views/4)

## This repository

This repository requires `git-lfs` to be able to check out images and binary objects.

Clone repo with:

```
git clone --recurse-submodules git@github.com:kitsudaiki/Hanami-AI.git
```

In case `git-lfs` while cloning and installed afterwards:

```
git lfs fetch --all
git lfs pull
```

In case the repo was cloned without submodules initially:

```
git submodule init
git submodule update --recursive
```

Mkdocs and plugins:

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

If you want to contribute things to this project, then I'm really happy about this. Please restrict this for the moment to bug-reports and feature-requests. Use the issue-templates for this. You can also place questions by these issues, if you want some information about parts of the project or if you want to try it out and need some help.
