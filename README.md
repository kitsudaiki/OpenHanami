# Hanami-AI

![Github workfloat status](https://img.shields.io/github/actions/workflow/status/kitsudaiki/Hanami-AI/build_test.yml?branch=develop&style=flat-square&label=build%20and%20test)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/kitsudaiki/Hanami-AI?label=version&style=flat-square)
![GitHub](https://img.shields.io/github/license/kitsudaiki/Hanami-AI?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Linux--x64-lightgrey?style=flat-square)

![Logo](assets/hanami-logo-with-text.png)

# **IMPORTANT: This project is still an experimental prototype at the moment and NOT ready for productive usage.** 

There is still a huge bunch of known bugs and missing validations, which can break the backend. Even the documentation here is quite basic. Normally I absolutely dislike it to make something public, which has known bugs and other problems, but I simply don't wanted to wait longer for the open-sourcing of this project. Most of it will be fixed until [Version `0.2.0`](/#roadmap). Keep in mind, that this project is created by a single person in his spare time beside a 40h/week job. ;)

## Intro

Hanami-AI is an AI-as-a-Service project, based on a concept created by myself. It is written from scratch with a Backend in C++ with Web-frontend.

The actual prototype consists of:

- partially implementation of an own concept for an artificial neuronal network. It has no fixed connections between the nodes, but creates connections over time while learning. Additionally it doesn't need a normalization of input-values and this way it can also handle unknown data as input. This should make it flexible and efficient. The current state is extremely experimental.
- very experimental but working GPU-support with CUDA and OpenCL
- multi-user- and multi-project-support, so multiple-users can share the same physical host
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

- To get a first impression there is a first example-workflow via the dashboard:

    https://docs.hanami-ai.com/How_To/2_dashboard/

- Even it is quite basic for now, there are also some internal workflow and tasks of the single components described:

    https://docs.hanami-ai.com/Inner_Workings/1_overview/

- Many basic dependencies were created in context of this project. Here is an overview of all involved repositories:

    https://docs.hanami-ai.com/other/1_dependencies/

## Core-components

for more details see [Documentation inner workings](/Inner_Workings/1_overview/)

- **Kyouko**
    - Content: Core-component, which holds the artificial neuronal networks.
    - prebuild Docker-Image: `kitsudaiki/kyouko_mind:0.2.0`

- **Misaki**
    - Content: Authentication-service and management of user
    - prebuild Docker-Image: `kitsudaiki/misaki_guard:0.2.0`

- **Shiori**
    - Content: Storage-component, which holds snapshots, logs and so on
    - prebuild Docker-Image: `kitsudaiki/shiori_archive:0.2.0`

- **Azuki**
    - Content: Monitoring and energy-optimization
    - prebuild Docker-Image: `kitsudaiki/azuki_heart:0.2.0`

- **Torii**
    - Content: Proxy for all incoming connections
    - prebuild Docker-Image: `kitsudaiki/torii_gateway:0.2.0`

- **Dashboard**
    - Content: Web-Frontend
    - prebuild Docker-Image: `kitsudaiki/hanami_ai_dashboard:0.2.0`

## Roadmap

- **0.1.0**
    - first prototype with basic feature-set

- **0.2.0**
    - merge all involved repositories into the main-repository
    - internal restructures, primary for the GPU-support
    - experimental GPU-support wiht CUDA and OpenCL (disabled at the moment)
    - general minor improvements

- **0.3.0**
    - *desired date*: Q3 2023
    - *content*: 
        - complete implementation of the core-concept and further evaluation and improvement of the learning-process:
            - allow to use it as spiking-neuronal-network
            - remove strict layer-structure, which is still enforced by hard configuration at the moment
            - build 3-dimensional networks
            - re-add the old reduction-process again
        - further evaluation and improving of the core-process
        - make GPU-support usable

- **0.4.0**
    - *desired date*: Q4 2023
    - *content*: 
        - first Multi-Node-Setup
        - rework dashboard


## Issue-Overview

[Hanami-AI-Project](https://github.com/users/kitsudaiki/projects/9/views/4)

## This repository

This repository requires `git-lfs` to be able to check out images and binary objects.

Clone repo with:

```
git clone --recurse-submodules git@github.com:kitsudaiki/Hanami-AI.git
```

In case `git-lfs` while cloning and installed afterwards:

```
git lfs fetch --all
```

In case the repo was cloned without submodules initially:

```
git submodule init
git submodule update --recursive
```


## Author

**Tobias Anker**

eMail: tobias.anker@kitsunemimi.moe

## License

The complete project is under [Apache 2 license](https://github.com/kitsudaiki/Hanami-AI/blob/develop/LICENSE).

## Contributing

If you want to contribute things to this project, then I'm really happy about this. Please restrict this for the moment to bug-reports and feature-requests. Use the issue-templates for this. You can also place questions by these issues, if you want some information about parts of the project or if you want to try it out and need some help.
