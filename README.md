# Hanami-AI

![Github workfloat status](https://img.shields.io/github/actions/workflow/status/kitsudaiki/Hanami-AI/build_test.yml?branch=develop&style=flat-square&label=build%20and%20test)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/kitsudaiki/Hanami-AI?label=version&style=flat-square)
![GitHub](https://img.shields.io/github/license/kitsudaiki/Hanami-AI?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Linux--x64-lightgrey?style=flat-square)

![Logo](assets/hanami-logo-with-text.png)

**IMPORTANT: This project is still an experimental prototype at the moment and NOT ready for productive usage.** 

There is still a huge bunch of known bugs and missing validations, which can break the backend. Even the documentation here is quite basic. Normally I absolutely dislike it to make something public, which has known bugs and other problems, but I simply don't wanted to wait longer for the open-sourcing of this project. Most of it will be fixed until [Version `0.2.0`](/#roadmap). Keep in mind, that this project is created by a single person in his spare time beside a 40h/week job. ;)

## Intro

Hanami-AI is an AI-as-a-Service project, based on a concept created by myself. It is written from scratch with a Backend in C++ with Web-frontend.

The actual prototype consists of:

- partially implementation of an own concept for an artificial neuronal network. It has no fixed connections between the nodes, but creates connections over time while learning. Additionally it doesn't need a normalization of input-values and this way it can also handle unknown data as input. This should make it flexible and efficient. The current state is extremely experimental.
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

- For the naming at some points look into the Glossar:

    https://docs.hanami-ai.com/other/2_glossar/

- Even it is quite basic for now, there are also some internal workflow and tasks of the single components described:

    https://docs.hanami-ai.com/Inner_Workings/1_overview/

- Many basic dependencies were created in context of this project. Here is an overview of all involved repositories:

    https://docs.hanami-ai.com/other/1_dependencies/

## Core-components

for more details see [Documentation inner workings](/Inner_Workings/1_overview/)

- **Kyouko**
    - Content: Core-component, which holds the artificial neuronal networks.
    - Repository: [KyoukoMind](https://github.com/kitsudaiki/KyoukoMind.git)
    - prebuild Docker-Image: `kitsudaiki/kyouko_mind:develop`

- **Misaki**
    - Content: Authentication-service and management of user
    - Repository: [MisakiGuard](https://github.com/kitsudaiki/MisakiGuard.git)
    - prebuild Docker-Image: `kitsudaiki/misaki_guard:develop`

- **Shiori**
    - Content: Storage-component, which holds snapshots, logs and so on
    - Repository: [ShioriArchive](https://github.com/kitsudaiki/ShioriArchive.git)
    - prebuild Docker-Image: `kitsudaiki/shiori_archive:develop`

- **Azuki**
    - Content: Monitoring and energy-optimization
    - Repository: [AzukiHeart](https://github.com/kitsudaiki/AzukiHeart.git)
    - prebuild Docker-Image: `kitsudaiki/azuki_heart:develop`

- **Torii**
    - Content: Proxy for all incoming connections
    - Repository: [ToriiGateway](https://github.com/kitsudaiki/ToriiGateway.git)
    - prebuild Docker-Image: `kitsudaiki/torii_gateway:develop`

- **Dashboard**
    - Content: Web-Frontend
    - Repository: [Dashboard](https://github.com/kitsudaiki/Hanami-AI-Dashboard.git)
    - prebuild Docker-Image: `kitsudaiki/hanami_ai_dashboard:develop`

## Roadmap

- [ ] improve documentation
- [ ] Open-API REST-documentation

- learning-process:
    - [ ] GPU-support
    - [ ] allow to use it as spiking-neuronal-network
    - [ ] remove strict layer-structure, which is still enforced by hard configuration at the moment
    - [ ] build 3-dimensional networks
    - [ ] add classical static neuronal networks

- [ ] Multi-Node-Setups

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

Tobias Anker

eMail: tobias.anker@kitsunemimi.moe

## License

The complete project is under Apache 2 license.

## Contributing

I'm happy, if you find the project interesting enough to contribute code. In this case, please wait until version `0.2.0`, because there are many API- and Database-breaking changes on the project. Additionally until `0.2.0` I will also provide a Code-Styling-Guide (at least for the C++-backend).
