# Dependencies

## Packages

Installed packages under the actual used Ubuntu 23.04

### Build

| apt-package | Purpose |
| --- | --- |
| clang-15 | C++-compiler |
| clang-format-15  | Helper for styling the source-code |
| make  | Build-Tool |
| qt5-qmake  | Qt-specific build-tool, because pro-files are used |
| bison  | Parser-Generator |
| flex  | Lexxer for the Parser-Generator |
| libssl-dev | ssl-library for TCS-encryption of network-connections |
| libcrypto++-dev | HMAC, SHA256 and other crypto related operations |
| libboost1.74-dev | Provides the Beast-library of Boost, which is used for the REST-API within the Torii |
| uuid-dev  | Generate UUID's within the code |
| libsqlite3-dev | Library to interact with the SQLite3 databases |
| protobuf-compiler | Convert protobuf-files into source-code |
| gcc | C-compiler |
| g++  | C++-compiler |
| nvidia-cuda-toolkit | Libraries and compiler for the CUDA-Kernel |

### Submodules

| name | License | Purpose |
| --- | --- | --- |
| Thalhammer/jwt-cpp | MIT | create and validate jwt-token |
| nlohmann/json | MIT | json-parser |

### Runtime

| apt-package | Purpose |
| --- | --- |
| openssl | ssl-library for TCS-encryption of network-connections | 
| libuuid1  | Generate UUID's within the code | 
| libcrypto++8  | HMAC, SHA256 and other crypto related operations | 
| libsqlite3-0  | Library to interact with the SQLite3 databases | 
| libprotobuf23 | Runtime-library for protobuffers | 
| libboost1.74 | Provides the Beast-library of Boost, which is used for the REST-API within the Torii |
| libcudart11.0 | Runtime-library for CUDA | 

## Overview

The following diagramm shows the basic relations of the library and tools with each other.

![Overview](../img/overview_dependencies.drawio)

I know, this is not a valid UML-diagram or something like this. It should old visualize the relations. A few connections in the diagram doesn't exist at the moment.


| Name | Description |
| --- | --- |
| **Hanami** | Core of the Project |
| **Hanami-Dashboard** | Web-Client to directly interact with the Hanami-instance |
| **hanamictl** | CLI-Client to directly interact with the Hanami-instance |
| **SDK_API_Testing** | Functional tests for SDK-library and REST-API | 
| **hanami_sdk** | SDK-library to provide functions for all supported actions to interact with the REST-API and automatic handling of the token-exchange. |
| **hanami_cluster_parser** | Parser-library for cluster-templates |
| **hanami_policies** | Parser for custon policy-files. |
| **hanami_hardware** | Collect and aggregate information of the local available hardware ressources. |
| **hanami_files** | File-handler and converter. |
| **hanami_network** | Self-created session-layer-protocol for network-communication in the Kitsunemimi-projects. |
| **hanami_database** | Abstration-layer for access databases. At the moment it only contains functionalities for easier creating of sql-requests. |
| **hanami_sqlite** | Simple wrapper-library for Sqlit3 databases. |
| **hanami_cpu** | Simple library to read different information of the cpu, like topological information, speed and energy consumption with RAPL, manipulate the speed of single cores of the cpu and read information of the local memory. |
| **hanami_obj** | This library provides a simple and minimal wavefront obj-parser and creator to generate the content of such files. |
| **hanami_opencl** | Simple wrapper-library for some commonly used OpenCL-functionallities. |
| **hanami_config** | Parser for ini-formated config files. |
| **hanami_args** | Small and easy to use parser for CLI-arguments. |
| **hanami_ini** | Parser for the content of ini-files. |
| **hanami_crypto** | Wrapper-library for crypto-operation from other external libraries, to simplify the usage of basic operation like AES, HMAC, SHA256, etc.  |
| **hanami_common** | Simple C++ library with commenly used functions for memory-handling, thread-handling, data representation and testing.  |
