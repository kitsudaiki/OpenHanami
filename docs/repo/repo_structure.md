# Repoistory structure

This sections should provide a basic overview of the repository and its components, in order to make
it easier for a new person to understand the code.

## General

    .
    ├── deploy
    │   ├── ansible
    │   └── k8s
    │
    ├── docs
    │
    ├── example_configs
    │
    ├── src
    │   ├── archive
    │   │   ├── frontend
    │   │   │   └── OpenHanami-Dashboard
    │   │   ├── libraries
    │   │   │   ├── hanami_network
    │   │   │   ├── hanami_obj
    │   │   │   └── hanami_opencl
    │   │   └── sdk
    │   │       └── javascript
    │   │
    │   ├── cli
    │   │   └── hanamictl
    │   │       ├── common
    │   │       └── resources
    │   │
    │   ├── hanami
    │   │   └── (see below)
    │   │
    │   ├── libraries
    │   │   ├── hanami_args
    │   │   ├── hanami_cluster_parser
    │   │   ├── hanami_common
    │   │   ├── hanami_config
    │   │   ├── hanami_cpu
    │   │   ├── hanami_crypto
    │   │   ├── hanami_database
    │   │   ├── hanami_hardware
    │   │   ├── hanami_ini
    │   │   ├── hanami_messages
    │   │   ├── hanami_policies
    │   │   └── hanami_sqlite
    │   │
    │   ├── sdk
    │   │   ├── go
    │   │   └── python
    │   │
    │   └── third-party-libs
    │       └── jwt-cpp
    │
    └── testing
        ├── ansible_deploy
        ├── go_cli_api
        └── python_sdk_api

-   **deploy**

    Contains the helm-chart doploying hanami on kubernetes and the ansible-playbook for a bare-metal
    installation.

-   **docs**

    Mkdocs-Documentation, where also this page here belongs to.

-   **example_configs**

    Example-configs for hanami. They are also used for tests within the CI-pipeline to make sure,
    that these examples are up-to-date.

-   **src**

    -   **archive**

        Old archived code, which is planned to be used or refactored again in the future. This was
        placed into the dedicated directory, because dead-code shouldn't be mixed within the rest.

    -   **cli**

        Code of the CLI-client written in Go

    -   **hanami**

        Contains the main-part of [hanami](/repo/repo_structure/#hanami-source-code)

    -   **libraries**

        -   **hanami_args**

            Provide argument-parser for the CLI of hanami

        -   **hanami_cluster_parser**

            Parser for the [Cluster-templates](/frontend/cluster_templates/cluster_template/)

        -   **hanami_common**

            Simple C++ library with commenly used functions for memory-handling, thread-handling,
            data representation and testing.

        -   **hanami_config**

            Parser for the config-file of hanami.

        -   **hanami_cpu**

            Simple library to read different information of the cpu, like topological information,
            speed and energy consumption with RAPL, manipulate the speed of single cores of the cpu
            and read information of the local memory.

        -   **hanami_crypto**

            Wrapper-library for crypto-operation from other external libraries, to simplify the
            usage of basic operation like AES, HMAC, SHA256, etc.

        -   **hanami_database**

            General abstraction-layer to interact with SQL-databases and creates SQL-queries out fo
            data-structure.

        -   **hanami_hardware**

            Collect hardware-information of the local system.

        -   **hanami_ini**

            ini-file parser, which is used by the config-library

        -   **hanami_messages**

            Protobuf-message definition

        -   **hanami_policies**

            Parser for the policy-file of hanami

        -   **hanami_sqlite**

            Functions to interact with sqlit3. Is used by the database-li

    -   **sdk**

        Code of the python SDK library and the go-version of the SDK-lib used by the CLI

    -   **third-party-libs**

        Third-party libraries as submodules. At the moment this is only the jwt-lib

-   **testing**

    Skripts for basic tests of the python SDK and the CLI. They are used within the CI-pipeline for
    basic tests of the components and the API.

## hanami source-code

    └── src
        └── hanami
            ├── src
            │   ├── api
            │   │   ├── endpoint_processing
            │   │   ├── http
            │   │   │   ├── auth
            │   │   │   ├── checkpoint
            │   │   │   ├── cluster
            │   │   │   ├── dataset
            │   │   │   ├── hosts
            │   │   │   ├── logs
            │   │   │   ├── measurements
            │   │   │   ├── project
            │   │   │   ├── system_info
            │   │   │   ├── task
            │   │   │   ├── threading
            │   │   │   └── user
            │   │   └── websocket
            │   ├── core
            │   │   ├── cluster
            │   │   ├── io
            │   │   │   ├── checkpoint
            │   │   │   └── data_set
            │   │   └── processing
            │   │       ├── cpu
            │   │       └── cuda
            │   ├── database
            │   └── documentation
            └── tests

-   **api**

    -   **endpoint_processing**

        General functions to provice a http-server and to process request over the REST-api like
        prechecks and so on, before triggering the endpoint-functions within the `http` directory

    -   **http**

        Definitions of all API-endpoints

    -   **websocket**

        Functions to send and receive data via websocket in case of a dataset-upload or direct
        interaction with a cluster

-   **core**

    -   **cluster**

        Datastruture, initializing and state-machine for the clusters and there data

    -   **io**

        All functions for serialization and deserialization of information like datasets and
        checkpoints

    -   **processing**

        Functions to process the data of a cluster

-   **database**

    Contains the definitions of the database-tables and all functions to interact with the
    SQL-database

-   **documentation**

    Code for generating the documenation (open-api docu, config docu, ...)
