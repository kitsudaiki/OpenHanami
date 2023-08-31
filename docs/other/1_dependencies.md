# Dependencies

## Packages

Installed packages under the actual used Ubuntu 22.04

### Build

| apt-package | Purpose |
| --- | --- |
| gcc | C-compiler |
| g++  | C++-compiler |
| make  | Build-Tool |
| qt5-qmake  | Qt-specific build-tool, because pro-files are used |
| bison  | Parser-Generator |
| flex  | Lexxer for the Parser-Generator |
| xxd | Transform file with OpenCL-Kernel into a string-variable within a header-file |
| libssl-dev | ssl-library for TCS-encryption of network-connections |
| libcrypto++-dev | HMAC, SHA256 and other crypto related operations |
| libboost1.74-dev | Provides the Beast-library of Boost, which is used for the REST-API within the Torii |
| uuid-dev  | Generate UUID's within the code |
| libsqlite3-dev | Library to interact with the SQLite3 databases |
| protobuf-compiler | Convert protobuf-files into source-code |
| ocl-icd-opencl-dev | Library for the OpenCL-Code |
| opencl-headers | Headers for OpenCL |
| nvidia-cuda-toolkit | Libraries and compiler for the CUDA-Kernel |


### Runtime

| apt-package | Purpose |
| --- | --- |
| openssl | ssl-library for TCS-encryption of network-connections | 
| libuuid1  | Generate UUID's within the code | 
| libcrypto++8  | HMAC, SHA256 and other crypto related operations | 
| libsqlite3-0  | Library to interact with the SQLite3 databases | 
| libprotobuf23 | Runtime-library for protobuffers | 
| libboost1.74 | Provides the Beast-library of Boost, which is used for the REST-API within the Torii |
| ocl-icd-libopencl1 | Runtime-library for OpenCL | 
| libcudart12.0 | Runtime-library for CUDA | 

## Overview

The following diagramm shows the basic relations of the library and tools with each other.

![Overview](../img/overview_dependencies.drawio)

I know, this is not a valid UML-diagram or something like this. It should old visualize the relations. A few connections in the diagram doesn't exist at the moment.

!!! info

    Originally there were all separated repositories, but in context of [issue #31](https://github.com/kitsudaiki/Hanami-AI/issues/31) all were packed into the main-repository of the project, in order to massivly reduce the maintenance workload.

### Components

[Hanami](#Hanami)

[Hanami-AI-Dashboard](#hanamiai-dashboard)

[SDK_API_Testing](#sdk_api_testing)

### Libraries

[libHanamiAiSdk](#libhanamiaisdk)

[libKitsunemimiHanamiClusterParser](#libkitsunemimihanamisegmentparser)

[libKitsunemimiHanamiPolicies](#libKitsunemimiHanamiPolicies)

[libKitsunemimiHanamiFiles](#libKitsunemimiHanamiFiles)

[libKitsunemimiHanamiHardware](#libKitsunemimiHanamiHardware)

[libKitsunemimiSakuraNetwork](#libkitsunemimisakuranetwork)

[libKitsunemimiSakuraDatabase](#libkitsunemimisakuradatabase)

[libKitsunemimiSakuraHardware](#libkitsunemimisakurahardware)

[libKitsunemimiSqlite](#libkitsunemimisqlite)

[libKitsunemimiCpu](#libkitsunemimicpu)

[libKitsunemimiObj](#libkitsunemimiobj)

[libKitsunemimiOpencl](#libkitsunemimiopencl)

[libKitsunemimiConfig](#libkitsunemimiconfig)

[libKitsunemimiArgs](#libkitsunemimiargs)

[libKitsunemimiNetwork](#libkitsunemiminetwork)

[libKitsunemimiIni](#libkitsunemimiini)

[libKitsunemimiJwt](#libkitsunemimijwt)

[libKitsunemimiCrypto](#libkitsunemimicrypto)

[libKitsunemimiJson](#libkitsunemimijson)

[libKitsunemimiCommon](#libKkitsunemimicommon)


??? question "Why the libraries are names `libKitsunemimi...`"

    Originally I searched for a name schema for the libraries to differentiation them from other libraries. For this and because my private domain was already `kitsunemimi.moe`, I decided to name my libraries `libKitsunemimi...`, because kitsunemimi are moe. ;) 

## Directories

__________

### Hanami

- **content**: Core of the Project

- **language**: `C++17`

__________

### Hanami-AI-Dashboard

- **content**: Web-Client to directly interact with the KyoukoMind-instance.

- **language**: `JavaScript + HTML + CSS`

__________

### SDK_API_Testing

- **content**: 
    - Functional tests for SDK-library and REST-API

- **language**: `C++17`

__________

### libHanamiAiSdk

- **content**: SDK-library to provide functions for all supported actions to interact with the REST-API and automatic handling of the token-exchange.

- **language**: `C++17`, `Javascript`

__________

### libKitsunemimiHanamiClusterParser

- **content**: Parser-library for cluster-templates

- **language**: `C++17`

__________

### libKitsunemimiHanamiPolicies

- **content**: Parser for custon policy-files.

- **language**: `C++17`

__________

### libKitsunemimiHanamiHardware

- **content**: Hardware related functions to regulate cpu and so on.

- **language**: `C++17`

__________

### libKitsunemimiHanamiFiles

- **content**: File-handler and converter.

- **language**: `C++17`

__________

### libKitsunemimiSakuraNetwork

- **content**: Self-created session-layer-protocol for network-communication in the Kitsunemimi-projects.

- **language**: `C++17`

__________

### libKitsunemimiSakuraDatabase

- **content**: Abstration-layer for access databases. At the moment it only contains functionalities for easier creating of sql-requests.

- **language**: `C++17`

__________

### libKitsunemimiSakuraHardware

- **content**: Collect and aggregate information of the local available hardware ressources.

- **language**: `C++17`

__________

### libKitsunemimiSqlite

- **content**: Simple wrapper-library for Sqlit3 databases.

- **language**: `C++17`

__________

### libKitsunemimiCpu

- **content**: Simple library to read different information of the cpu, like topological information, speed and energy consumption with RAPL, manipulate the speed of single cores of the cpu and read information of the local memory.

- **language**: `C++17`

__________

### libKitsunemimiStorage

- **content**: Small library to collect information of the local storage.

- **language**: `C++17`

__________

### libKitsunemimiInterface

- **content**: Small library to collect information of the local network-interfaces.

- **language**: `C++17`

__________

### libKitsunemimiObj

- **content**: This library provides a simple and minimal wavefront obj-parser and creator to generate the content of such files.

- **language**: `C++17`

__________

### libKitsunemimiOpencl

- **content**: Simple wrapper-library for some commonly used OpenCL-functionallities.

- **language**: `C++17`

__________

### libKitsunemimiConfig

- **content**: Parser for ini-formated config files.

- **language**: `C++17`

__________

### libKitsunemimiArgs

- **content**: Small and easy to use parser for CLI-arguments.

- **language**: `C++17`

__________

### libKitsunemimiNetwork

- **content**: This is a small library for network connections. It provides servers and clients for unix-domain-sockets, tcp-sockets and ssl encrypted tcp-sockets.

- **language**: `C++17`

__________

### libKitsunemimiIni

- **content**: Parser for the content of ini-files.

- **language**: `C++17`

__________

### libKitsunemimiJwt

- **content**: Library to create and validate JWT-tokens.

- **language**: `C++17`

__________

### libKitsunemimiCrypto

- **content**: Wrapper-library for crypto-operation from other external libraries, to simplify the usage of basic operation like AES, HMAC, SHA256, etc. 

- **language**: `C++17`

__________

### libKitsunemimiJson

- **content**: Parser for the content of json-files.

- **language**: `C++17`

__________

### libKitsunemimiCommon 

- **content**: Simple C++ library with commenly used functions for memory-handling, thread-handling, data representation and testing. 

- **language**: `C++17`
