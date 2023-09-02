# Hanami

![Logo](img/hanami-logo-with-text.png)


!!! danger "IMPORTANT"

    **This project is still an experimental prototype at the moment and NOT ready for productive usage.** 

    There is still a huge bunch of known bugs and missing validations, which can break the backend. Even the documentation here is quite basic. Normally I absolutely dislike it to make something public, which has known bugs and other problems, but I simply don't wanted to wait longer for the open-sourcing of this project. Most of it will be fixed until [Version `0.2.0`](/#roadmap). Keep in mind, that this project is created by a single person in his spare time beside a 40h/week job. ;)

!!! info

    At this documentation here at the moment is the develop-branch version. So this site will follow the develop-branches of the single components. Additionally there will be continuously updates of things, which are still missing in general in this documentation, like config-descriptions, sequence diagrams, manual to build docker-images and so on.

## Intro

Hanami is basically an AI-as-a-Service project, based on a concept created by myself. It is written from scratch with a Backend in C++ with Web-frontend.

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

## Basics

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Getting Started__

    ---

    Installation-Guide to deploy HanamiAI on a kubernetes

    [:octicons-arrow-right-24: Installation](/How_To/1_installation/)



-   :material-monitor-dashboard:{ .lg .middle } __First Look__

    ---

    To get a first impression there is a first example-workflow via the dashboard. 

    For the naming at some points look into the [Glossar](/other/2_glossar)

    [:octicons-arrow-right-24: Dashboard](/How_To/2_dashboard/)


-   :material-file-document-multiple-outline:{ .lg .middle } __OpenAPI Documentation__

    ---

    Automatic generated OpenAPI documentation

    [:octicons-arrow-right-24: OpenAPI documentation](/Resource_Docus/1_rest_api_documentation/)


-   :octicons-package-dependencies-24:{ .lg .middle } __Dependencies__

    ---

    Many basic dependencies were created in context of this project. Here is an overview of all involved repositories.

    [:octicons-arrow-right-24: Dependencies](/other/1_dependencies/)

</div>

## Issue-Overview

[Hanami-Project](https://github.com/users/kitsudaiki/projects/9/views/4)

## Author

**Tobias Anker**

eMail: tobias.anker@kitsunemimi.moe

## License

The complete project is under [Apache 2 license](https://github.com/kitsudaiki/Hanami/blob/develop/LICENSE).
