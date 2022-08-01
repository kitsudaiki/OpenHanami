# Dependencies

## Overview

The following diagramm shows the basic relations of the library and tools with each other.

<p align="center">
  <img src="pictures/overview.png?raw=true" alt="Overview"/>
</p>

## Actual repositories and dependencies

### Project core repositories

[KyoukoMind](#KyoukoMind)

[AzukiHeart](#AzukiHeart)

[MisakiGuard](#MisakiGuard)

[SagiriArchive](#SagiriArchive)

[IzumiShare](#IzumiShare)

[InoriLink](#InoriLink)

[ToriiGateway](#ToriiGateway)

[TsugumiTester](#TsugumiTester)

[libKyoukoMind](#libKyoukoMind)

[libAzukiHeart](#libAzukiHeart)

[libMisakiGuard](#libMisakiGuard)

[libSagiriArchive](#libSagiriArchive)

[libIzumiShare](#libIzumiShare)

[libInoriLink](#libInoriLink)

[libToriiGateway](#libToriiGateway)

### Project frontend repositories

[KitsumiAI-Dashboard](#KitsumiAI-Dashboard)

[KitsumiAI-CLI](#KitsumiAI-CLI)

[libKitsumiAiSdk](#libKitsumiAiSdk)

### hanami-layer libraries

[libKitsunemimiHanamiMessaging](#libKitsunemimiHanamiMessaging)

[libKitsunemimiHanamiPolicies](#libKitsunemimiHanamiPolicies)

[libKitsunemimiHanamiDatabase](#libKitsunemimiHanamiDatabase)

[libKitsunemimiHanamiEndpoints](#libKitsunemimiHanamiEndpoints)

[libKitsunemimiHanamiCommon](#libKitsunemimiHanamiCommon)

### sakura-layer libraries

[libKitsunemimiSakuraNetwork](#libKitsunemimiSakuraNetwork)

[libKitsunemimiSakuraLang](#libKitsunemimiSakuraLang)

[libKitsunemimiSakuraDatabase](#libKitsunemimiSakuraDatabase)

[libKitsunemimiSakuraHardware](#libKitsunemimiSakuraHardware)

### generic libraries

[libKitsunemimiSqlite](#libKitsunemimiSqlite)

[libKitsunemimiCpu](#libKitsunemimiCpu)

[libKitsunemimiStorage](#libKitsunemimiStorage)

[libKitsunemimiInterface](#libKitsunemimiInterface)

[libKitsunemimiObj](#libKitsunemimiObj)

[libKitsunemimiOpencl](#libKitsunemimiOpencl)

[libKitsunemimiConfig](#libKitsunemimiConfig)

[libKitsunemimiArgs](#libKitsunemimiArgs)

[libKitsunemimiNetwork](#libKitsunemimiNetwork)

[libKitsunemimiJinja2](#libKitsunemimiJinja2)

[libKitsunemimiIni](#libKitsunemimiIni)

[libKitsunemimiJwt](#libKitsunemimiJwt)

[libKitsunemimiCrypto](#libKitsunemimiCrypto)

[libKitsunemimiJson](#libKitsunemimiJson)

[libKitsunemimiCommon](#libKitsunemimiCommon)


## Repositories

__________

### KyoukoMind

#### Metadata

- **content**: Provides an artificial neural network based on a concept created by myself. Since version 0.4.0 it also has some influences of the commonly used deep-learning concept. 
Core characteristics:
    - No fully meshed random connections between nodes at the beginning. All connections are only created while learning new information.
    - No strict layer structure (layer-like structures are only optional).
    - No limitation for to the range [0.0, 1.0] for input- and output-values.

- **additional commentary**: Actual tests with the MNIST handwritten digits dataset came up to 98.1% correct matches.

- **current version**: `0.8.1`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.
xxd | xxd | >= 1.10 | converts text files into source code files, which is used to complile the OpenCl-kernel file into the source-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases
opencl-headers  | opencl-headers | 2.x | Header-files for opencl
ocl-icd-opencl-dev | ocl-icd-opencl-dev | 2.x | libraries for opencl

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiObj | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiObj.git
libKitsunemimiOpencl | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiOpencl.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSqlite | v0.3.0 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiSakuraDatabase | v0.5.0 |  -
libKitsunemimiHanamiCommon | v0.2.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiDatabase | v0.3.0 |  -
libKitsunemimiHanamiMessaging | v0.4.1 |  -
libKitsumiAiSdk | v0.3.1 | -
libAzukiHeart | v0.2.0 | -
libMisakiGuard | v0.1.0 | -
ibSagiriArchive | v0.2.0 | -

__________

### AzukiHeart

#### Metadata

- **content**: 
    - Ressource-management for all component in order to reduce the energyconsumption of the system
    - Monitoring to keep the system stable

- **current version**: `0.2.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiCpu | v0.3.0 |  -
libKitsunemimiSakuraNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiSakuraHardware | v0.1.1 |  -
libKitsunemimiHanamiCommon | v0.1.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiMessaging | v0.4.1 |  -
libAzukiHeart | v0.2.0 | - 
libMisakiGuard | v0.1.0 | - 

__________

### MisakiGuard

#### Metadata

- **content**: 
    - User-management with credentials, roles and policies
    - Create and validate JWT-Token
    - Automatic generation of user-specific REST-API-documentations for all components at runtime

- **current version**: `0.2.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSqlite | v0.3.0 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiSakuraDatabase | v0.5.0 |  -
libKitsunemimiHanamiCommon | v0.2.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiDatabase | v0.3.0 |  -
libKitsunemimiHanamiPolicies | v0.1.0 |  -
libKitsunemimiHanamiMessaging | v0.4.1 |  -
libAzukiHeart | v0.2.0 | -
libMisakiGuard | v0.1.0 | -

__________

### SagiriArchive

#### Metadata

- **content**: 
    - Handling for all persisted objects in the backend (train-data, snapshots, etc.)
    - Central logging
        - Error-log
        - Audit-log

- **current version**: `0.3.1`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSqlite | v0.3.0 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiSakuraDatabase | v0.5.0 |  -
libKitsunemimiHanamiCommon | v0.2.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiDatabase | v0.3.0 |  -
libKitsunemimiHanamiMessaging | v0.4.1 |  -
libAzukiHeart | v0.2.0 | -
libMisakiGuard | v0.1.0 | -
ibSagiriArchive | v0.2.0 |  -

__________

### IzumiShare

#### Metadata

- **current version**: `-`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.3 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSqlite | v0.3.0 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiSakuraDatabase | v0.4.1 |  -
libKitsunemimiHanamiCommon | v0.1.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiDatabase | v0.2.0 |  -
libKitsunemimiHanamiMessaging | v0.3.0 |  -
libAzukiHeart | v0.1.0 | -
libMisakiGuard | v0.1.0 | -

__________

### InoriLink

#### Metadata

- **current version**: `-`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.3 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSqlite | v0.3.0 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiSakuraDatabase | v0.4.1 |  -
libKitsunemimiHanamiCommon | v0.1.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiDatabase | v0.2.0 |  -
libKitsunemimiHanamiMessaging | v0.3.0 |  -
libAzukiHeart | v0.1.0 | -
libMisakiGuard | v0.1.0 | -

__________

### ToriiGateway

#### Metadata

- **content**: Proxy for networking communication between the components.

- **current version**: `0.6.1`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
uuid | uuid-dev | >= 2.34 | generate uuid's
boost-library | libboost1.71-dev | >= 1.71 | provides boost beast library for HTTP and Websocket client

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.2.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiMessaging | v0.4.1 |  -
libAzukiHeart | v0.2.0 |  -
ibSagiriArchive | v0.2.0 |  -

__________

### TsugumiTester

#### Metadata

- **content**: 
    - Functional tests for SDK-library, REST-API and CLI-tool
    - Benchmark tests

- **current version**: `0.3.1`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
boost-library | libboost1.71-dev | >= 1.71 | provides boost beast library for HTTP and Websocket client
uuid | uuid-dev | >= 2.34 | generate uuid's

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsumiAiSdk | v0.3.1 | -
libKitsunemimiHanamiCommon | v0.2.0 | -

__________

### libKyoukoMind

#### Metadata

- **content**: Lib for internal interaction with Kyouko

- **current version**: `-`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.3 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.1.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiMessaging | v0.3.0 |  -

__________

### libAzukiHeart

#### Metadata

- **content**: Lib for internal interaction with Azuki

- **current version**: `0.2.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.2.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiMessaging | v0.4.1 |  -

__________

### libMisakiGuard

#### Metadata

- **content**: Lib for internal interaction with Misaki

- **current version**: `0.1.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.3 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.1.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiMessaging | v0.3.0 |  -

__________

### libSagiriArchive

#### Metadata

- **content**: Lib for internal interaction with Sagiri

- **current version**: `0.2.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.2.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiMessaging | v0.4.0 |  -

__________

### libIzumi...

#### Metadata

- **content**: Lib for internal interaction with Izumi

- **current version**: `-`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.3 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.1.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiMessaging | v0.3.0 |  -

__________

### libInori...

#### Metadata

- **content**: Lib for internal interaction with Inori

- **current version**: `-`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.3 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.1.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiMessaging | v0.3.0 |  -

__________

### libToriiGateway

#### Metadata

- **content**: Lib for internal interaction with the Torii

- **current version**: `-`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.3 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.1.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -
libKitsunemimiHanamiMessaging | v0.3.0 |  -

__________

### KitsumiAI-Dashboard

#### Metadata

- **content**: Web-Client to directly interact with the KyoukoMind-instance.

- **current version**: -

- **language**: `JavaScript + HTML + CSS`

- **visibility**: `private`

- **location**: `private gitlab`

__________

### KitsumiAI-CLI

#### Metadata

- **content**: Cli-Client to directly interact with the KyoukoMind-instance.

- **current version**: -

- **language**: `go`

- **visibility**: `private`

- **location**: `private gitlab`

#### required tools to build

name | repository | version | task
--- | --- | --- | ---
go | golang | >= 1.13 | Compiler for the go code.

#### Required generic libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
subcommands | 1.2.0 | https://github.com/kitsudaiki/subcommands (fork of https://github.com/google/subcommands)
tablewriter | 0.0.5 | https://github.com/kitsudaiki/tablewriter (fork of https://github.com/olekukonko/tablewriter)

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsumiAiSdk | v0.1.0 | -

__________

### SakuraTree

(This project is actually paused.)

SakuraTree provides an automation tool to deploy applications, with high performance thanks to some features like easy parallelism of tasks and a self-created file syntax. It was primary created for the components of the Kyouko-Project and beside this also to automate different tasks on my deployment at home.

Documentation of current version: https://files.kitsunemimi.moe/docs/SakuraTree-Documentation_0_4_1.pdf

#### Metadata

- **content**: This is/should become a simple-to-use and fast automation tool to deploy tools and files on multiple nodes.

- **current version**: `0.4.1`

- **license**: `Apache 2`

- **language**: `C++14`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/SakuraTree.git

#### required tools to build

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 6.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
boost-filesystem library | libboost-filesystem-dev | 1.6x | interactions with files and directories on the system

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.15.1 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiPersistence | v0.10.0 | https://github.com/kitsudaiki/libKitsunemimiPersistence.git
libKitsunemimiArgs | v0.2.1 | https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiJson | v0.10.4 | https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.8.0 | https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.4.5 | https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiSakuraLang | v0.5.1 | https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git

__________

### libKitsumiAiSdk

#### Metadata

- **content**: SDK-library to provide functions for all supported actions to interact with the REST-API and automatic handling of the token-exchange.

- **current version**: `0.3.1`

- **language**: `C++17`, `go`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools for C++-part

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries for C++-part

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES
boost-library | libboost1.71-dev | >= 1.71 | provides boost beast library for HTTP and Websocket client
uuid | uuid-dev | >= 2.34 | generate uuid's

#### Required kitsunemimi libraries for C++-part

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiHanamiCommon | v0.2.0 | -


#### Required build tools for go-part

name | repository | version | task
--- | --- | --- | ---
go | golang | >= 1.13 | Compiler for the go code.

__________

### libKitsunemimiHanamiMessaging

#### Metadata

- **content**: Additional application-layer of the project related network stack.

- **current version**: `0.4.1`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections
uuid | uuid-dev | >= 2.34 | generate uuid's
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.8.2 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiArgs | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJwt | v0.4.1 |  -
libKitsunemimiSakuraNetwork | v0.8.4 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiHanamiCommon | v0.2.0 |  -
libKitsunemimiHanamiEndpoints | v0.1.0 |  -

__________

### libKitsunemimiHanamiPolicies

#### Metadata

- **content**: Parser for custon policy-files.

- **current version**: `0.1.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
uuid | uuid-dev | >= 2.34 | generate uuid's

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.1 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiIni | v0.5.1 | https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiArgs | v0.4.0 | https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 | https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiHanamiCommon | v0.1.0 | -

__________

### libKitsunemimiHanamiDatabase

#### Metadata

- **content**: Add user and project scroped handling of database-entries

- **current version**: `0.3.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
uuid | uuid-dev | >= 2.34 | generate uuid's
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.0 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 | https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiSqlite | v0.3.0 | -
libKitsunemimiSakuraDatabase | v0.5.0 | -

__________

### libKitsunemimiHanamiEndpoints

#### Metadata

- **content**: Parser-library for custom endpoint-definitions

- **current version**: `0.1.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
uuid | uuid-dev | >= 2.34 | generate uuid's

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.1 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiIni | v0.5.1 | https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiArgs | v0.4.0 | https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 | https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiHanamiCommon | v0.1.0 | -

__________

### libKitsunemimiHanamiCommon

#### Metadata

- **content**: Common library for the Hanami-Layer

- **current version**: `0.2.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `private gitlab`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
uuid | uuid-dev | >= 2.34 | generate uuid's

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.0|  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiIni | v0.5.1 | https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiArgs | v0.4.0 | https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.4.0 | https://github.com/kitsudaiki/libKitsunemimiConfig.git

__________

### libKitsunemimiSakuraNetwork

#### Metadata

- **content**: Self-created session-layer-protocol for network-communication in the Kitsunemimi-projects.

- **current version**: `0.8.4`

- **license**: `Apache 2`

- **language**: `C++17`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | 1.1.x | encryption for tls connections

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.3 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiNetwork | v0.8.2 | https://github.com/kitsudaiki/libKitsunemimiNetwork.git

__________

### libKitsunemimiSakuraLang

#### Metadata

- **content**: The library `libKitsunemimiSakuraLang` provides a simple script-language created by myself. It is packed as library for easy used in different tools. Originally it was created exclusively for the SakuraTree project (https://github.com/kitsudaiki/SakuraTree), but in the end it become generic and flexible enough to be also interesting for other upcoming projects, so it was moved into its own library.

- **current version**: `0.12.0`

- **license**: `Apache 2`

- **language**: `C++17`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git

#### required tools to build

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | >= 3.0 | Build the parser-code together with the lexer-code.

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.0 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.3 | https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.9.1 | https://github.com/kitsudaiki/libKitsunemimiJinja2.git

__________

### libKitsunemimiSakuraDatabase

#### Metadata

- **content**: Abstration-layer for access databases. At the moment it only contains functionalities for easier creating of sql-requests.

- **current version**: `0.5.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `github`

#### required tools to build

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases
uuid | uuid-dev | >= 2.30 | generate uuid's

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.0 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiSqlite | v0.3.0 | -

__________

### libKitsunemimiSakuraHardware

#### Metadata

- **content**: Collect and aggregate information of the local available hardware ressources.

- **current version**: `0.1.1`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `github`

#### required tools to build

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.26.1 | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiCpu | v0.3.0 | -

__________

### libKitsunemimiSqlite

#### Metadata

- **content**: Simple wrapper-library for Sqlit3 databases.

- **current version**: `0.3.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `github`

#### required tools to build

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
sqlite3 library | libsqlite3-dev | >= 3.0 | handling of sqlite databases

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.0 | https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiCpu

#### Metadata

- **content**: Simple library to read different information of the cpu, like topological information, speed and energy consumption with RAPL, manipulate the speed of single cores of the cpu and read information of the local memory.

- **current version**: `0.3.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `github`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

#### Other requirements for reading energy-consumption

- for CPUs of AMD Zen/Zen2 Linux-Kernel of version `5.8` or newer must be used, for Zen3 Linux-Kernel of version `5.11` or newer
- the `msr` kernel-module must be available and loaded

__________

### libKitsunemimiStorage

#### Metadata

- **content**: Small library to collect information of the local storage.

- **current version**: `0.1.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `github`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiInterface

#### Metadata

- **content**: Small library to collect information of the local network-interfaces.

- **current version**: `0.1.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `github`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiObj

#### Metadata

- **content**: This library provides a simple and minimal wavefront obj-parser and creator to generate the content of such files.

- **current version**: `0.2.0`

- **license**: `MIT`

- **language**: `C++17`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiObj.git

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.23.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiOpencl

#### Metadata

- **content**: Simple wrapper-library for some commonly used OpenCL-functionallities.

- **current version**: `0.4.0`

- **license**: `Apache 2`

- **language**: `C++17`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiOpencl.git

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
opencl-headers  | opencl-headers | 2.x | Header-files for opencl
ocl-icd-opencl-dev | ocl-icd-opencl-dev | 2.x | libraries for opencl

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.23.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiConfig

#### Metadata

- **content**: Parser for ini-formated config files.

- **current version**: `0.4.0`

- **license**: `MIT`

- **language**: `C++17`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiConfig.git

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | 3.x | Build the parser-code together with the lexer-code.

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.23.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiIni | v0.5.0 | https://github.com/kitsudaiki/libKitsunemimiIni.git

__________

### libKitsunemimiArgs

#### Metadata

- **content**: Small and easy to use parser for CLI-arguments.

- **current version**: `0.4.0`

- **license**: `MIT`

- **language**: `C++17`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiArgs.git

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.23.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiNetwork

#### Metadata

- **content**: This is a small library for network connections. It provides servers and clients for unix-domain-sockets, tcp-sockets and ssl encrypted tcp-sockets.

- **current version**: `0.8.2`

- **license**: `MIT`

- **language**: `C++17`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiNetwork.git

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

#### Required generic libraries

name | repository | version | task
--- | --- | --- | ---
ssl library | libssl-dev | >= 1.1.1f | encryption for tls connections

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.23.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiJinja2

#### Metadata

- **content**: Simple but imcomplete converter for jinja2-templates.

- **current version**: `0.9.1`

- **license**: `MIT`

- **language**: `C++17`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiJinja2.git

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | 3.x | Build the parser-code together with the lexer-code.

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.24.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiJson | v0.11.2 |  https://github.com/kitsudaiki/libKitsunemimiJson.git

__________

### libKitsunemimiIni

#### Metadata

- **content**: Parser for the content of ini-files.

- **current version**: `0.5.1`

- **license**: `MIT`

- **language**: `C++17`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiIni.git

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | 3.x | Build the parser-code together with the lexer-code.

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.24.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiJwt

#### Metadata

- **content**: Library to create and validate JWT-tokens.

- **current version**: `0.4.1`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `github`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | 3.x | Build the parser-code together with the lexer-code.
ssl library | libssl-dev | >= 1.1.1f | provides signing-functions
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiCrypto | v0.2.0 |  -
libKitsunemimiJson | v0.11.3 |  https://github.com/kitsudaiki/libKitsunemimiJson.git

__________

### libKitsunemimiCrypto

#### Metadata

- **content**: Wrapper-library for crypto-operation from other external libraries, to simplify the usage of basic operation like AES, HMAC, SHA256, etc. 

- **current version**: `0.2.0`

- **language**: `C++17`

- **visibility**: `private`

- **location**: `github`

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
ssl library | libssl-dev | >= 1.1.1f | provides signing-functions
crpyto++ | libcrypto++-dev | >= 5.6 | provides encryption-functions like AES

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.23.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiJson

#### Metadata

- **content**: Parser for the content of json-files.

- **current version**: `0.11.3`

- **license**: `MIT`

- **language**: `C++17`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiJson.git

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | 3.x | Build the parser-code together with the lexer-code.

#### Required kitsunemimi libraries

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | v0.25.0 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git

__________

### libKitsunemimiCommon 

#### Metadata

- **content**: Simple C++ library with commenly used functions for memory-handling, thread-handling, data representation and testing. 

- **current version**: `0.26.1`

- **license**: `MIT`

- **language**: `C++17`

- **visibility**: `public`

- **location**: `github`

- **repo-path**: https://github.com/kitsudaiki/libKitsunemimiCommon.git

#### Required build tools

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.

