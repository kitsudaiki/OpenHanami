# Dependencies

## Overview

The following diagramm shows the basic relations of the library and tools with each other.

![Overview](../img/overview_dependencies.drawio)

I know, this is not a valid UML-diagram or something like this. It should old visualize the relations. A few connections in the diagram doesn't exist at the moment.

!!! warning
    
    This overview is at the moment not fully in sync.

### Hanami-AI specific repositories

[KyoukoMind](#kyoukomind)

[AzukiHeart](#azukiheart)

[MisakiGuard](#misakiguard)

[ShioriArchive](#shioriarchive)

[ToriiGateway](#toriigateway)

[TsugumiTester](#tsugumitester)

[Hanami-AI-Dashboard](#hanamiai-dashboard)

[libAzukiHeart](#libazukiheart)

[libMisakiGuard](#libmisakiguard)

[libShioriArchive](#libshioriarchive)

[libHanamiAiSdk](#libhanamiaisdk)

### Hanami-Layer

[libKitsunemimiHanamiNetwork](#libkitsunemimihanaminetwork)

[libKitsunemimiHanamiPolicies](#libkitsunemimihanamipolicies)

[libKitsunemimiHanamiDatabase](#libkitsunemimihanamidatabase)

[libKitsunemimiHanamiEndpoints](#libkitsunemimihanamiendpoints)

[libKitsunemimiHanamiCommon](#libkitsunemimihanamicommon)

### Sakura-Layer

[libKitsunemimiSakuraNetwork](#libkitsunemimisakuranetwork)

[libKitsunemimiSakuraLang](#libkitsunemimisakuralang)

[libKitsunemimiSakuraDatabase](#libkitsunemimisakuradatabase)

[libKitsunemimiSakuraHardware](#libkitsunemimisakurahardware)

### Common libraries

These simple generic libraries with wrapper, parser and functionalities I often use. Most of these stuff like CLI-argument-parser and so on, already exist in various implementations on github, but I wanded to create my own versions to have maximum control over the requirements and to have only a minimal set of funtions, which I really need.

[libKitsunemimiSqlite](#libkitsunemimisqlite)

[libKitsunemimiCpu](#libkitsunemimicpu)

[libKitsunemimiStorage](#libkitsunemimistorage)

[libKitsunemimiInterface](#libkitsunemimiinterface)

[libKitsunemimiObj](#libkitsunemimiobj)

[libKitsunemimiOpencl](#libkitsunemimiopencl)

[libKitsunemimiConfig](#libkitsunemimiconfig)

[libKitsunemimiArgs](#libkitsunemimiargs)

[libKitsunemimiNetwork](#libkitsunemiminetwork)

[libKitsunemimiJinja2](#libkitsunemimijinja2)

[libKitsunemimiIni](#libkitsunemimiini)

[libKitsunemimiJwt](#libkitsunemimijwt)

[libKitsunemimiCrypto](#libkitsunemimicrypto)

[libKitsunemimiJson](#libkitsunemimijson)

[libKitsunemimiCommon](#libKkitsunemimicommon)


??? question "Why the libraries are names `libKitsunemimi...`"

    Originally I searched for a name schema for the libraries to differentiation them from other libraries. For this and because my private domain was already `kitsunemimi.moe`, I decided to name my libraries `libKitsunemimi...`, because kitsunemimi are moe. ;) 

## Repositories

__________

### KyoukoMind

**Metadata**

- **content**: Provides an artificial neural network based on a concept created by myself. Since version 0.4.0 it also has some influences of the commonly used deep-learning concept. 
Core characteristics:
    - No fully meshed random connections between nodes at the beginning. All connections are only created while learning new information.
    - No strict layer structure (layer-like structures are only optional).
    - No limitation for to the range [0.0, 1.0] for input- and output-values.

- **additional commentary**: Actual tests with the MNIST handwritten digits dataset came up to 98.1% correct matches.

- **current version**: `0.9.1`

- **language**: `C++17`

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

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.10.0 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.6.0 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiJwt | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiJwt.git
libKitsunemimiSqlite | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiSqlite.git
libKitsunemimiSakuraNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.13.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiSakuraDatabase | v0.6.1 |  https://github.com/kitsudaiki/libKitsunemimiSakuraDatabase.git
libKitsunemimiHanamiCommon | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git
libKitsunemimiHanamiEndpoints | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiEndpoints.git
libKitsunemimiHanamiDatabase | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiDatabase.git
libKitsunemimiHanamiNetwork | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiNetwork.git
libAzukiHeart | v0.3.0 |  https://github.com/kitsudaiki/libAzukiHeart.git
ibShioriArchive | v0.3.0 |  https://github.com/kitsudaiki/ibShioriArchive.git
libMisakiGuard | v0.2.0 | https://github.com/kitsudaiki/ibShioriArchive.git

__________

### AzukiHeart

**Metadata**

- **content**: 
    - Ressource-management for all component in order to reduce the energyconsumption of the system
    - Monitoring to keep the system stable

- **current version**: `0.3.1`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.10.0 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.6.0 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiJwt | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiJwt.git
libKitsunemimiCpu | v0.4.1 |  https://github.com/kitsudaiki/libKitsunemimiCpu.git
libKitsunemimiSakuraNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.13.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiSakuraHardware | v0.6.1 |  https://github.com/kitsudaiki/libKitsunemimiSakuraHardware.git
libKitsunemimiHanamiCommon | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git
libKitsunemimiHanamiEndpoints | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiEndpoints.git
libKitsunemimiHanamiNetwork | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiNetwork.git
libAzukiHeart | v0.3.0 |  https://github.com/kitsudaiki/libAzukiHeart.git
libMisakiGuard | v0.2.0 | https://github.com/kitsudaiki/ibShioriArchive.git

__________

### MisakiGuard

**Metadata**

- **content**: 
    - User-management with credentials, roles and policies
    - Create and validate JWT-Token
    - Automatic generation of user-specific REST-API-documentations for all components at runtime

- **current version**: `0.3.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.10.0 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.6.0 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiJwt | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiJwt.git
libKitsunemimiSqlite | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiSqlite.git
libKitsunemimiSakuraNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.13.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiSakuraDatabase | v0.6.1 |  https://github.com/kitsudaiki/libKitsunemimiSakuraDatabase.git
libKitsunemimiHanamiCommon | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git
libKitsunemimiHanamiEndpoints | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiEndpoints.git
libKitsunemimiHanamiDatabase | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiDatabase.git
libKitsunemimiHanamiNetwork | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiNetwork.git
libKitsunemimiHanamiPolicies | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiPolicies.git
libAzukiHeart | v0.3.0 |  https://github.com/kitsudaiki/libAzukiHeart.git
ibShioriArchive | v0.3.0 |  https://github.com/kitsudaiki/ibShioriArchive.git
libMisakiGuard | v0.2.0 | https://github.com/kitsudaiki/ibShioriArchive.git

__________

### ShioriArchive

**Metadata**

- **content**: 
    - Handling for all persisted objects in the backend (train-data, snapshots, etc.)
    - Central logging
        - Error-log
        - Audit-log

- **current version**: `0.4.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.10.0 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.6.0 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiJwt | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiJwt.git
libKitsunemimiSqlite | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiSqlite.git
libKitsunemimiSakuraNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.13.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiSakuraDatabase | v0.6.1 |  https://github.com/kitsudaiki/libKitsunemimiSakuraDatabase.git
libKitsunemimiHanamiCommon | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git
libKitsunemimiHanamiEndpoints | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiEndpoints.git
libKitsunemimiHanamiDatabase | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiDatabase.git
libKitsunemimiHanamiNetwork | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiNetwork.git
libAzukiHeart | v0.3.0 |  https://github.com/kitsudaiki/libAzukiHeart.git
ibShioriArchive | v0.3.0 |  https://github.com/kitsudaiki/ibShioriArchive.git
libMisakiGuard | v0.2.0 | https://github.com/kitsudaiki/ibShioriArchive.git

__________

### ToriiGateway

**Metadata**

- **content**: Proxy for networking communication between the components.

- **current version**: `0.7.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's
boost-library | libboost1.71-dev | >= 1.71 | provides boost beast library for HTTP and Websocket client

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.10.0 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.6.0 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiJwt | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiJwt.git
libKitsunemimiSakuraNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.13.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git
libKitsunemimiHanamiEndpoints | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiEndpoints.git
libKitsunemimiHanamiNetwork | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiNetwork.git
libAzukiHeart | v0.3.0 |  https://github.com/kitsudaiki/libAzukiHeart.git
ibShioriArchive | v0.3.0 |  https://github.com/kitsudaiki/ibShioriArchive.git

__________

### TsugumiTester

**Metadata**

- **content**: 
    - Functional tests for SDK-library, REST-API and CLI-tool
    - Benchmark tests

- **current version**: `0.3.1`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
boost-library | libboost1.71-dev | >= 1.71 | provides boost beast library for HTTP and Websocket client
uuid | uuid-dev | >= 2.34 | generate uuid's

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libHanamiAiSdk | v0.3.1 | -
libKitsunemimiHanamiCommon | v0.2.0 | -

__________

### libAzukiHeart

**Metadata**

- **content**: Lib for internal interaction with Azuki

- **current version**: `0.3.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.10.0 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.6.0 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiJwt | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiJwt.git
libKitsunemimiSakuraNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.13.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git
libKitsunemimiHanamiEndpoints | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiEndpoints.git
libKitsunemimiHanamiNetwork | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiNetwork.git

__________

### libMisakiGuard

**Metadata**

- **content**: Lib for internal interaction with Misaki

- **current version**: `0.2.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.10.0 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.6.0 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiJwt | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiJwt.git
libKitsunemimiSakuraNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.13.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git
libKitsunemimiHanamiEndpoints | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiEndpoints.git
libKitsunemimiHanamiNetwork | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiNetwork.git
__________

### libShioriArchive

**Metadata**

- **content**: Lib for internal interaction with Shiori

- **current version**: `0.3.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.10.0 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.6.0 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiJwt | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiJwt.git
libKitsunemimiSakuraNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.13.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git
libKitsunemimiHanamiEndpoints | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiEndpoints.git
libKitsunemimiHanamiNetwork | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiNetwork.git

__________

### Hanami-AI-Dashboard

**Metadata**

- **content**: Web-Client to directly interact with the KyoukoMind-instance.

- **current version**: `0.2.0`

- **language**: `JavaScript + HTML + CSS`

__________

### libHanamiAiSdk

**Metadata**

- **content**: SDK-library to provide functions for all supported actions to interact with the REST-API and automatic handling of the token-exchange.

- **current version**: `0.4.0`

- **language**: `C++17`, `go`


**Required build tools** for C++-part

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries** for C++-part

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
boost-library | libboost1.71-dev | >= 1.71 | provides boost beast library for HTTP and Websocket client
uuid | uuid-dev | >= 2.34 | generate uuid's

**Required kitsunemimi libraries** for C++-part

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiHanamiCommon | v0.3.0 | https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git


**Required build tools** for go-part

name | repository | version | task
--- | --- | --- | ---
go | golang | >= 1.13 | Compiler for the go code.

__________

### libKitsunemimiHanamiNetwork

**Metadata**

- **content**: Additional application-layer of the project related network stack.

- **current version**: `0.5.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.10.0 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.6.0 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiJwt | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiJwt.git
libKitsunemimiSakuraNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.13.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git
libKitsunemimiHanamiEndpoints | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiEndpoints.git

__________

### libKitsunemimiHanamiPolicies

**Metadata**

- **content**: Parser for custon policy-files.

- **current version**: `0.1.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
uuid | uuid-dev | >= 2.34 | generate uuid's

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.1 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiIni | v0.5.1 | https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiArgs | v0.4.0 | https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 | https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiHanamiCommon | v0.1.0 | -

__________

### libKitsunemimiHanamiDatabase

**Metadata**

- **content**: Add user and project scroped handling of database-entries

- **current version**: `0.4.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
uuid | uuid-dev | >= 2.34 | generate uuid's
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 | https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiSqlite | v0.4.0 | https://github.com/kitsudaiki/libKitsunemimiSqlite.git
libKitsunemimiSakuraDatabase | v0.6.1 | https://github.com/kitsudaiki/libKitsunemimiSakuraDatabase.git
libKitsunemimiHanamiCommon | v0.3.0 | https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git

__________

### libKitsunemimiHanamiEndpoints

**Metadata**

- **content**: Parser-library for custom endpoint-definitions

- **current version**: `0.2.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
uuid | uuid-dev | >= 2.34 | generate uuid's

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiIni | v0.6.0 | https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiArgs | v0.5.0 | https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 | https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiHanamiCommon | v0.3.0 | https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git

__________

### libKitsunemimiHanamiCommon

**Metadata**

- **content**: Common library for the Hanami-Layer

- **current version**: `0.3.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
uuid | uuid-dev | >= 2.34 | generate uuid's

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiIni | v0.6.0 | https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiArgs | v0.5.0 | https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 | https://github.com/kitsudaiki/libKitsunemimiConfig.git

__________

### libKitsunemimiSakuraNetwork

**Metadata**

- **content**: Self-created session-layer-protocol for network-communication in the Kitsunemimi-projects.

- **current version**: `0.9.0`

- **license**: `Apache 2`

- **language**: `C++17`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiNetwork | v0.9.0 | https://github.com/kitsudaiki/libKitsunemimiNetwork.git

__________

### libKitsunemimiSakuraLang

**Metadata**

- **content**: The library `libKitsunemimiSakuraLang` provides a simple script-language created by myself. It is packed as library for easy used in different tools. Originally it was created exclusively for the SakuraTree project (https://github.com/kitsudaiki/SakuraTree), but in the end it become generic and flexible enough to be also interesting for other upcoming projects, so it was moved into its own library.

- **current version**: `0.13.0`

- **license**: `Apache 2`

- **language**: `C++17`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git

**Required tools to build**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.10.0 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git

__________

### libKitsunemimiSakuraDatabase

**Metadata**

- **content**: Abstration-layer for access databases. At the moment it only contains functionalities for easier creating of sql-requests.

- **current version**: `0.6.1`

- **language**: `C++17`

**Required tools to build**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases
uuid | uuid-dev | >= 2.30 | generate uuid's

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 | https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiSqlite | v0.4.0 | https://github.com/kitsudaiki/libKitsunemimiSqlite.git

__________

### libKitsunemimiSakuraHardware

**Metadata**

- **content**: Collect and aggregate information of the local available hardware ressources.

- **current version**: `0.2.0`

- **language**: `C++17`

**Required tools to build**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiCpu | v0.4.1 | https://github.com/kitsudaiki/libKitsunemimiCpu.git

__________

### libKitsunemimiSqlite

**Metadata**

- **content**: Simple wrapper-library for Sqlit3 databases.

- **current version**: `0.4.0`

- **language**: `C++17`

**Required tools to build**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson| v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git


__________

### libKitsunemimiCpu

**Metadata**

- **content**: Simple library to read different information of the cpu, like topological information, speed and energy consumption with RAPL, manipulate the speed of single cores of the cpu and read information of the local memory.

- **current version**: `0.4.1`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

**Other requirements for reading energy-consumption**

- for CPUs of AMD Zen/Zen2 Linux-Kernel of version `5.8` or newer must be used, for Zen3 Linux-Kernel of version `5.11` or newer (NOT TESTED FOR AMD)
- the `msr` kernel-module must be available and loaded

__________

### libKitsunemimiStorage

**Metadata**

- **content**: Small library to collect information of the local storage.

- **current version**: `0.1.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiInterface

**Metadata**

- **content**: Small library to collect information of the local network-interfaces.

- **current version**: `0.1.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiObj

**Metadata**

- **content**: This library provides a simple and minimal wavefront obj-parser and creator to generate the content of such files.

- **current version**: `0.2.0`

- **license**: `MIT`

- **language**: `C++17`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiObj.git

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.23.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiOpencl

**Metadata**

- **content**: Simple wrapper-library for some commonly used OpenCL-functionallities.

- **current version**: `0.4.0`

- **license**: `Apache 2`

- **language**: `C++17`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiOpencl.git

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
opencl-headers  | opencl-headers | 2.x | Header-files for opencl
ocl-icd-opencl-dev | ocl-icd-opencl-dev | 2.x | libraries for opencl

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.23.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiConfig

**Metadata**

- **content**: Parser for ini-formated config files.

- **current version**: `0.5.0`

- **license**: `MIT`

- **language**: `C++17`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiConfig.git

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | 3.x | Build the parser-code together with the lexer-code.

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiIni | v0.6.0 | https://github.com/kitsudaiki/libKitsunemimiIni.git

__________

### libKitsunemimiArgs

**Metadata**

- **content**: Small and easy to use parser for CLI-arguments.

- **current version**: `0.5.0`

- **license**: `MIT`

- **language**: `C++17`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiArgs.git

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiNetwork

**Metadata**

- **content**: This is a small library for network connections. It provides servers and clients for unix-domain-sockets, tcp-sockets and ssl encrypted tcp-sockets.

- **current version**: `0.9.0`

- **license**: `MIT`

- **language**: `C++17`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiNetwork.git

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

**Required generic libraries**

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | >= 1.1.1f | encryption for tls connections

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiJinja2

**Metadata**

- **content**: Simple but imcomplete converter for jinja2-templates.

- **current version**: `0.10.0`

- **license**: `MIT`

- **language**: `C++17`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiJinja2.git

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | 3.x | Build the parser-code together with the lexer-code.

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git

__________

### libKitsunemimiIni

**Metadata**

- **content**: Parser for the content of ini-files.

- **current version**: `0.6.0`

- **license**: `MIT`

- **language**: `C++17`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiIni.git

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | 3.x | Build the parser-code together with the lexer-code.

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiJwt

**Metadata**

- **content**: Library to create and validate JWT-tokens.

- **current version**: `0.5.1`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | 3.x | Build the parser-code together with the lexer-code.
ssl library | libssl-dev | >= 1.1.1f | provides signing-functions
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git

__________

### libKitsunemimiCrypto

**Metadata**

- **content**: Wrapper-library for crypto-operation from other external libraries, to simplify the usage of basic operation like AES, HMAC, SHA256, etc. 

- **current version**: `0.2.0`

- **language**: `C++17`

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
ssl library | libssl-dev | >= 1.1.1f | provides signing-functions
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiJson

**Metadata**

- **content**: Parser for the content of json-files.

- **current version**: `v0.12.0`

- **license**: `MIT`

- **language**: `C++17`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiJson.git

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | 3.x | Build the parser-code together with the lexer-code.

**Required kitsunemimi libraries**

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiCommon 

**Metadata**

- **content**: Simple C++ library with commenly used functions for memory-handling, thread-handling, data representation and testing. 

- **current version**: `0.27.1`

- **license**: `MIT`

- **language**: `C++17`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiCommon.git

**Required build tools**

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

