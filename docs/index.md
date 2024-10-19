!!! info

    This is the documentation of the develop-branch.

# OpenHanami

<p align="center">
  <img src="img/hanami-logo-with-text.png" width="500" height="594" />
</p>

!!! danger "IMPORTANT"

    # **IMPORTANT: his project is still an experimental prototype and NOT ready for any productive usage. There are still many missing tests, input-validations and so on. Beside this there is also still quite a lot of evaluation and improving of the current features necessary.**

## Intro

OpenHanami contains in its core a custom experimental artificial neural network, which can work on
unnormalized and unfiltered input-data, like sensor measurement data. The network growth over time
by creating new nodes and connections between the nodes while learning new data. The base concept
was created by myself and the code was written from scratch without any frameworks. The goal behind
OpenHanami is to create something unique, which works more like the human brain. It wasn't targeted
to get a higher accuracy than classical artificial neural networks like Tensorflow, but to be more
flexible and easier to use and more efficient in resource-consumption for big amounts of inputs and
users. Additionally it also provides an as-a-Service architecture within a cloud native environment
and multi-tenancy.

## Current prototypically implemented features

-   **Growing neural network**:

    The artificial neural network, which is the core of the project, growth over time while learning
    new things by creating new nodes and connections between the nodes based on the given input. A
    resize of the network is also quite linear in complexity.

-   **No normalization of input**

    The input of the network is not restricted to range of 0.0 - 1.0 . Every value, as long it is a
    positive value, can be inserted. Also if there is a single broken value in the input-data, which
    is million times higher, than the rest of the input-values, it has nearly no effect on the rest
    of the already trained data. Thanks to the reduction-process, all synapses, which are only the
    result of this single input, are removed again from the network.

-   **Reduction-Process**

    The concept of a growing network has the result, that there is basically nearly no limit in
    size, even if the growth-rate slows down over time. To limit the growth-rate even more, it is
    possible to enable a reduction-process, which removes synapses again, which were to inactive to
    reach the threshold to be marked as persistent.

    See
    [measurement-examples](https://docs.openhanami.com/inner_workings/measurements/measurements/#reduction_1)

-   **Spiking neural network**

    The concept also supports a special version of working as a spiking neural network. This is
    optional for a created network and basically has the result, that an input is impacted by an
    older input, based on the time how long ago this input happened.

    See
    [short explanation](https://docs.openhanami.com/inner_workings/core/core/#spiking-neural-network)
    and [measurement-examples](https://docs.openhanami.com/inner_workings/measurements/measurements)

-   **No strict layer structure**

    The base of a new neural network is defined by a cluster-template. In these templates the
    structure of the network in planed in hexagons, indeed of layer. When a node tries to create a
    new synapse, the location of the target-node depends on the location of the source-node within
    these hexagons. The target is random and the probability depends on the distance to the source.
    This way it is possible to break the static layer structure. But when defining a line of
    hexagons and allow nodes only to connect to the nodes of the next hexagon, a classical
    layer-structure can still be enforced.

    See
    [short explanation](https://docs.openhanami.com/inner_workings/core/core/#no-strict-layer-structure)
    and [measurement-examples](https://docs.openhanami.com/inner_workings/measurements/measurements)

-   **3-dimensional networks**

    It is basically possible to define 3-dimensional networks. This was only added, because the
    human brain is also a 3D-object. This feature exist in the
    [cluster-templates](https://docs.openhanami.com/frontend/cluster_templates/cluster_template/),
    but was never tested until now. Maybe in bigger tests in the future this feature could become
    useful to better mix information with each other.

-   **Parallelism**

    The processing structure works also for multiple threads, which can work at the same time on the
    same network. (GPU-support with CUDA is disabled at the moment for various reasons).

-   **Usable performance**

    The 60.000 training pictures of the MNIST handwritten letters can be trained on CPU in about 3
    seconds for the first epoch, without any batch-processing of the input-data and results in an
    accuracy of 91-93 % after this time.

-   **Generated OpenAPI-Documentation**

    The OpenAPI-documentation is generated directly from the code. So changing the settings of a
    single endpoint in the code automatically results in changes of the resulting documentation, to
    make sure, that code and documentation are in sync.

    See [OpenAPI-docu](https://docs.openhanami.com/frontend/rest_api_documentation/)

-   **Multi-user and multi-project**

    The projects supports multiple user and multiple projects with different roles (member,
    project-admin and admin) and also managing the access to single api-endpoints via policy-file.
    Each user can login by username and passphrase and gets an JWT-token to access the user- and
    project-specific resources.

    See [Authentication-docu](https://docs.openhanami.com/inner_workings/user_and_projects/)

-   **Efficient resource-usage**

    1. The concept of the neural network results in the effect, that only necessary synapses of an
       active node of the network is processed, based on the input. So if only very few input-nodes
       get data pushed in, there is less processing-time necessary to process the network.

    2. Because of the multi-user support, multiple networks of multiple users can be processed on
       the same physical host and share the RAM, CPU-cores and even the GPU, without splitting them
       via virtual machines or vCPUs.

    3. Capability to regulate the cpu-frequencey and measure power-consumption. (disabled currently)

        See
        [Monitoring-docu](https://docs.openhanami.com/inner_workings/monitoring/monitoring/#controlling-cpu-frequency)

-   **Network-input**

    There are 2-variants, how it interact with the neural networks:

    1. Uploading the dataset and starting an asynchronous task based on this dataset over the API

        See [OpenAPI-docu](https://docs.openhanami.com/frontend/rest_api_documentation/)

    2. Directly communicate with the neural network via websocket. In this case not a whole dataset
       is push through the synapse, but instead only a single network-input is send. The call is
       blocking, until the network returns the output, which gives more control.

        See [Websocket-docu](https://docs.openhanami.com/frontend/websockets/websocket_workflow/)

-   **Installation on Kubernetes**

    The backend can be basically deployed on kubernetes via Helm-chart.

    See [Installation-docu](https://docs.openhanami.com/backend/installation/)

## Known disadvantages

The concept is not perfekt and also has some disadvantages, which are the result of the architecture
itself:

-   Very inefficient for binary input, where inputs of single nodes of the network can only be 0 or
    1, without values in between

-   A single synapse needs more memory than in a classical network. The hope is, in bigger tests, it
    becomes much more efficient compared to fully meshed layered networks.

## Summary important links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Getting started**

    ***

    [:octicons-arrow-right-24: Example-Workflow](/frontend/example_workflow/)

    [:octicons-arrow-right-24: Installation-Guide](/backend/installation/)

    [:octicons-arrow-right-24: SDK and CLI documentation](/frontend/cli_sdk_docu/)

    [:octicons-arrow-right-24: OpenAPI documentation](/frontend/rest_api_documentation/)

-   :octicons-codespaces-24:{ .lg .middle } **Development**

    ***

    [:octicons-arrow-right-24: How to build](/repo/build_guide/)

    [:octicons-arrow-right-24: Development-Guide](/repo/development/)

    [:octicons-arrow-right-24: Contributing guide](https://github.com/kitsudaiki/OpenHanami/blob/develop/CONTRIBUTING.md)

    [:octicons-arrow-right-24: Dependency-Overview](/repo/dependencies/)

-   :octicons-package-24:{ .lg .middle } **Pre-build objects**

    ***

    All objects are automatically build and uploaded by the
    [CI-pipeline](https://github.com/kitsudaiki/OpenHanami/actions/workflows/build_test.yml) for
    each merge on `develop`-branch and for each tag.

    [:octicons-arrow-right-24: Docker-images](https://hub.docker.com/repository/docker/kitsudaiki/hanami/tags)

    [:octicons-arrow-right-24: client, SDK and helm-chart](https://files.openhanami.com/)

-   :octicons-milestone-24:{ .lg .middle } **Roadmap**

    ***

    [:octicons-arrow-right-24: Roadmap](/home/ROADMAP/)

</div>

## Currently disabled features

There are some features, which existed in the past, were disabled temporary and will be
added/enabled again in the near future:

1. Dashboard

    As a PoC a first dashboard was created, without any framework. It is planned to refactor this
    old version in `v0.9.0` and re-write it again with Typescript and some additional frameworks.
    Until then, it is temporary disabled, because it would current cost too much time to keep this
    unused and prototypical version up-to-data. As reference see the example-workflow of the
    PoC-dashboard: [Dashboard-docu](https://docs.openhanami.com/frontend/dashboard/dashboard/)

2. Regulation of CPU-speed

    Also in older version there also was the function available to regulate the speed of the CPU
    based on the workload. The dashboard was used to visualize the CPU metrics like the speed. Since
    the dashboard was disabled, there is at the moment not feedback available, so for usability
    reasons the feature was not further maintained and disabled for now. It is planned again for
    version `v0.8.0`.

3. GPU-support

    There already were some attempts in the past to add GPU-support with CUDA and OpenCL in the
    past. Some version like 0.4.0 also had a working version implemented. The problem was
    disappointing performance and some restrictions for the CPU-version too. There will be some
    further attempts in the future, to fix this issue and bring GPU support back into the project,
    but because there is no definite solution now, it is unknown when this happens.

## Author

**Tobias Anker**

eMail: tobias.anker@kitsunemimi.moe

## License

The complete project is under
[Apache 2 license](https://github.com/kitsudaiki/OpenHanami/blob/develop/LICENSE).
