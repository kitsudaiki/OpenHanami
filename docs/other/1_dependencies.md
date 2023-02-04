# Dependencies

## Overview

The following diagramm shows the basic relations of the library and tools with each other.

![Overview](../img/overview_dependencies.drawio)

I know, this is not a valid UML-diagram or something like this. It should old visualize the relations. A few connections in the diagram doesn't exist at the moment.

!!! info

    Originally there were all separated repositories, but in context of [issue #31](https://github.com/kitsudaiki/Hanami-AI/issues/31) all were packed into the main-repository of the project, in order to massivly reduce the maintenance workload.

### Components

[KyoukoMind](#kyoukomind)

[AzukiHeart](#azukiheart)

[MisakiGuard](#misakiguard)

[ShioriArchive](#shioriarchive)

[ToriiGateway](#toriigateway)

[TsugumiTester](#tsugumitester)

[Hanami-AI-Dashboard](#hanamiai-dashboard)

### Libraries

[libAzukiHeart](#libazukiheart)

[libMisakiGuard](#libmisakiguard)

[libShioriArchive](#libshioriarchive)

[libHanamiAiSdk](#libhanamiaisdk)

[libKitsunemimiHanamiClusterParser](#libkitsunemimihanamisegmentparser)

[libKitsunemimiHanamiSegmentParser](#libkitsunemimihanamiclusterparser)

[libKitsunemimiHanamiNetwork](#libkitsunemimihanaminetwork)

[libKitsunemimiHanamiPolicies](#libkitsunemimihanamipolicies)

[libKitsunemimiHanamiDatabase](#libkitsunemimihanamidatabase)

[libKitsunemimiHanamiCommon](#libkitsunemimihanamicommon)

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

## Build-requirements

### C++

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.
xxd | xxd | >= 1.10 | converts text files into source code files, which is used to complile the OpenCl-kernel file into the source-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases
opencl-headers  | opencl-headers | 2.x | Header-files for opencl
ocl-icd-opencl-dev | ocl-icd-opencl-dev | 2.x | libraries for opencl

## Directories

__________

### KyoukoMind

- **content**: Provides an artificial neural network based on a concept created by myself. Since version 0.4.0 it also has some influences of the commonly used deep-learning concept. 
Core characteristics:
    - No fully meshed random connections between nodes at the beginning. All connections are only created while learning new information.
    - No strict layer structure (layer-like structures are only optional).
    - No limitation for to the range [0.0, 1.0] for input- and output-values.

- **language**: `C++17`

__________

### AzukiHeart

- **content**: 
    - Ressource-management for all component in order to reduce the energyconsumption of the system
    - Monitoring to keep the system stable

- **language**: `C++17`
__________

### MisakiGuard

- **content**: 
    - User-management with credentials, roles and policies
    - Create and validate JWT-Token
    - Automatic generation of user-specific REST-API-documentations for all components at runtime

- **language**: `C++17`

__________

### ShioriArchive

- **content**: 
    - Handling for all persisted objects in the backend (train-data, snapshots, etc.)
    - Central logging
        - Error-log
        - Audit-log

- **language**: `C++17`

__________

### ToriiGateway

- **content**: Proxy for networking communication between the components.

- **language**: `C++17`

__________

### TsugumiTester

- **content**: 
    - Functional tests for SDK-library, REST-API and CLI-tool
    - Benchmark tests

- **language**: `C++17`

__________

### libAzukiHeart

- **content**: Lib for internal interaction with Azuki

- **language**: `C++17`

__________

### libMisakiGuard

- **content**: Lib for internal interaction with Misaki

- **language**: `C++17`

__________

### libShioriArchive

- **content**: Lib for internal interaction with Shiori

- **language**: `C++17`

__________

### Hanami-AI-Dashboard

- **content**: Web-Client to directly interact with the KyoukoMind-instance.

- **language**: `JavaScript + HTML + CSS`

__________

### libHanamiAiSdk

- **content**: SDK-library to provide functions for all supported actions to interact with the REST-API and automatic handling of the token-exchange.

- **language**: `C++17`, `Javascript`

__________

### libKitsunemimiHanamiClusterParser

- **content**: Parser-library for cluster-templates

- **language**: `C++17`

__________

### libKitsunemimiHanamiSegmentParser

- **content**: Parser-library for segment-templates

- **language**: `C++17`

__________

### libKitsunemimiHanamiNetwork

- **content**: Additional application-layer of the project related network stack.

- **language**: `C++17`

__________

### libKitsunemimiHanamiPolicies

- **content**: Parser for custon policy-files.

- **language**: `C++17`

__________

### libKitsunemimiHanamiDatabase

- **content**: Add user and project scroped handling of database-entries

- **language**: `C++17`

__________

### libKitsunemimiHanamiCommon

- **content**: Common library for the Hanami-Layer

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
