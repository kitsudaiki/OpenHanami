# Hanami

<p align="center">
  <img src="img/hanami-logo-with-text.jpg" width="500" height="594" />
</p>


!!! danger "IMPORTANT"

    # **This project is still an experimental prototype at the moment and NOT ready for productive usage.** 

## Intro

Hanami contains in its core a custom concept for neural networks, which are very flexible in their behavior and structure, which is packed in an as-a-Service backend. The backend is completely written from scratch in C++.

## Initial goal

I started this project without a specific use-case in mind. The only goal was to create a neural network, which is much more dynamic, compared to other classical networks, in regard of learning behavior and structure. It should behave more like the human brain. So it was basically a private research project.
But it also shouldn't be only a simple PoC. I also wanted a good performance, so I written it in C++, optimized the data-structure many time and added multi-threading and CUDA-support.
Mostly it was tested so far with the MNIST-test, because this is the Hello-world of neural networks, but also other little constructed examples.

Despite the missing goal of the concept, this entire project was and still is a very good base for me to improve my developer skills and learn new technologies.

## Current state of the project

Like already written above, it is still a prototype. There are still many missing tests, input-validations, comments and so on. This project is currently only written by a single person beside a 40h/week job. 

## Current prototypically implemented features

- **Growing neural network**:

    The neural network, which is the core of the project, growth over time while learning new things by creating new synapses to other nodes, if the input requires this. A resize of the network is also quite linear in complexity.

- **Reduction-Process**

    The concept of a growing network has the result, that there is basically nearly no limit in size, even if the growth-rate slows down over time. To limit the growth-rate even more, it is possible to enable a reduction-process, which removes synapses again, which were to inactive to reach the threshold to be markes as persistent.

    See [measurement-examples](/inner_workings/measurements/measurements/#reduction_1)

- **No normalization of input**

    The input of the network is not restricted to range of 0.0 - 1.0 . Every value, as long it is a positive value, can be inserted. Also if there is a single broken value in the input-data, which is million times higher, than the rest of the input-values, it has nearly no effect on the rest of the already trained data. Thanks to the redcution-process, all synapses, which are only the result of this single input, are removed again from the network.

- **Spiking neural network**

    The concept also supports a special version of working as a spiking neural network. This is optional for a created network and basically has the result, that an input is impaced by an older input, based on the time how long ago this input happened.

    See [short explanation](/inner_workings/core/core/#spiking-neural-network) and  [measurement-examples](/inner_workings/measurements/measurements)

- **No strict layer structure**

    The base of a new neural network is defined by a cluster-template. In these templates the structure of the network in planed in hexagons, indeed of layer. When a node tries to create a new synapse, the location of the target-node depends on the location of the source-node within these hexagons. The target is random and the probability depends on the distance to the source. This way it is possible to break the static layer structure. 
    But when defining a line of hexagons and allow nodes only to connect to the nodes of the next hexagon, a classical layer-structure can still be enforced. 

    See [short explanation](/inner_workings/core/core/#no-strict-layer-structure) and  [measurement-examples](/inner_workings/measurements/measurements)

- **3-dimensional networks**

    It is basically possible to define 3-dimensional networks. This was only added, because the human brain is also a 3D-object. This feature exist in the [cluster-templates](/frontend/cluster_templates/cluster_template/), but was never tested until now. Maybe in bigger tests in the future this feature could become useful to better mix information with each other. 

- **Parallelism**

    The processing structure works also for multiple threads, which can work at the same time on the same network, and also works on gpu. 
    For the gpu only CUDA is suppored at the moment. With the next version it is also planned to port the CUDA-kernel to OpenCL. OpenCL was already supported in the past, but in the recent developing-process replaced by CUDA, because it was easier to keep up-to-date with the fast changing data-structures, than OpenCL.

- **Usable performance**

    The 60.000 training pictures of the MNIST handwritten letters can be trained on CPU in about 4 seconds for the first epoch, without any batch-processing of the input-data and results in an accuracy of 93-94 % after this time.

- **Generated OpenAPI-Documentattion**

    The OpenAPI-documentattion is generated directly from the code. So changing the settings of a single endpoint in the code automatically results in changes of the resulting documentation, to make sure, that code and documentattion are in sync.

    See [OpenAPI-docu](/frontend/rest_api_documentation/)

- **Multi-user and multi-project**

    The projecs supports multiple user and multiple projects with different roles (member, project-admin and admin) and also managing the access to single api-endpoints via policy-file. Each user can login by username and password and gets an JWT-token to access the user- and project-specific resources.

    See [Authentication-docu](/inner_workings/user_and_projects/)

- **Efficient resource-usage**

    1. The concept of the neural network results in the effect, that only necessary synapses of an active node of the network is processed, based on the input. So if only very few input-nodes get data pushed in, there is less processing-time necessary to process the network. 

    2. Because of the multi-user support, multiple networks of multiple users can be processed on the same physical host and share the RAM, CPU-cores and even the GPU, without spliting them via virtual machines or vCPUs.

    3. Capability to regulate the cpu-frequest and measure power-consuption. (disabled currently)

        See [Monitoring-docu](/inner_workings/monitoring/monitoring/#controlling-cpu-frequency)

- **Network-input**

    There are 2-variants, how it interact with the neural networks:

    1. Uploading the dataset and starting an asynchronous task based on this dataset over the API

        See [OpenAPI-docu](/frontend/rest_api_documentation/)

    2. Directly communicate with the neural network via websocket. In this case not a whole dataset is push through the synapse, but instead only a single network-input is send. The call is blocking, until the network returns the output, which gives more control.

        See [Websocket-docu](/frontend/websockets/websocket_workflow/)

- **Remote-connection**

    1. SDK (Python, Go, Javascript)

        The SDK for the project was implemented in Python, Go and Javascript. Most the the endpoints are covered by all 3 of them. The one with the most features is the Python-version. It is also the only one with the feature of direct communication with the network and is also the one used by the test-script, which tests all important endpoints.

        See [SDK-docu](/frontend/sdk_library/)

    2. CLI (Go)

        A first basic CLI-client also exist, which supports most the endpoints and is written in Go. In my opinion Go is really good for this, because of its standard library and because the binary are mostly statically linked, which results in less dependencies on the host, where it should be executed. Also I wanted to get some pracitcal experience to learn Go. The client uses the Go-version of the SDK.

        See [CLI-docu](/frontend/cli_client/)

    3. Dashboard (HTML + Javascript + CSS)

        As a PoC a first dashboard was created, without any framework. At the moment it is not really maintained, because of a lack of available time and uses the Javascript-version of the SDK. The main motivation for thsi dashboard was to learn basics of web-development and because a visual output simply looks better for showing this project to someone.

        See [Dashboard-docu](/frontend/dashboard/dashboard/)

- **Installation on Kubernetes and with Ansible**

    The backend can be basically deployed on kubernetes via Helm-chart or plain via Ansible.

    See [Installation-docu](/backend/installation/)

## Known disadvantages (so far)

- Very inefficient for binary input, where inputs of single nodes of the network can only be 0 or 1, without values in between

- A single synapse needs more memory than in a classical network. The hope is, in bigger tests, it becomes much more efficient compared to fully meshed layered networks.

- few inputs, but many outputs brings bad results. There ideas, hob to fix this issue, but these are not implemented and tested, if they are a good fix for this issue.

## Possible use-case (maybe)

Because the normalization of input is not necessary, together with the good performance of training single inputs (based on the benchmark) and the direct interaction remotely over websockets, could make this project useful for processing measurement-data of sensors of different machines, especially for new sensors, where the exact maximum output-values are unknown. So continuous training of the network right from the beginning would be possible, without collecting a bunch of data at first.

## Summary important links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Development__

    ---

    Short guide to prepare a local environment for developoment

    [:octicons-arrow-right-24: Development](/backend/development/)


-   :material-clock-fast:{ .lg .middle } __SDK and client__

    ---

    [:octicons-arrow-right-24: Python-SDK](/frontend/sdk_library/)

    [:octicons-arrow-right-24: CLI-client](/frontend/cli_client/)


-   :material-file-document-multiple-outline:{ .lg .middle } __OpenAPI Documentation__

    ---

    Automatic generated OpenAPI documentation

    [:octicons-arrow-right-24: OpenAPI documentation](/frontend/rest_api_documentation/)

-   :octicons-package-dependencies-24:{ .lg .middle } __Dependencies__

    ---

    Many basic dependencies were created in context of this project. Here is an overview of all involved repositories.

    [:octicons-arrow-right-24: Dependencies](/other/dependencies/)

-   :material-clock-fast:{ .lg .middle } __Installation__

    ---

    Installation-Guide to deploy Hanami on a Kubernetes or with Ansible

    [:octicons-arrow-right-24: Installation](/backend/installation/)


</div>

## Issue-Overview

[Hanami-Project](https://github.com/users/kitsudaiki/projects/9/views/4)

## Contributing

If you want to help the project by contributing things, you can check the [Contributing guide](https://github.com/kitsudaiki/Hanami/blob/develop/CONTRIBUTING.md).

## Author

**Tobias Anker**

eMail: tobias.anker@kitsunemimi.moe

## License

The complete project is under [Apache 2 license](https://github.com/kitsudaiki/Hanami/blob/develop/LICENSE).
