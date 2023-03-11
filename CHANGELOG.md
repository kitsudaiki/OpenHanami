# Changelog

## [Unreleased]

### Breaking-Changes

#### API-Breaking

- use now custom syntax for cluster- and segment-templates instead of json
- use protobuf-messages to send payload of files
- merge endpoints to create learn- and request-tasks for csv- and mnist-files into one endpoint
- creating a task doesn't require the explicit selecting of the data-set type. This is now read from the data-set metadata

#### Snapshot-Breaking

- changed internal data-structure of the core in order to make GPU-support possible without race-conditions

### Changed

- **merged all repositories (source-code, documentation, build, deploy) into this main-repository here**
- made token-key and registry in helm-chart configurable
- made token expire-time configurable in Misaki
- core-internal renamings of structs and functions
- use dropdown-menu instead of single buttons in table-entries within the dashboard
- use nested namespaces in sakura- and hanami-layer
- changed a huge amount of map-iterators to c++17 style

### Added 

- Added experimental GPU-support with CUDA and OpenCL (per default hard in source-code disbabled at the moment)
- internal tokens requestable form Misaki
- API-endpoints and database-table to request logs with page-selector
- print audit-logs in dashboard

### Fixed

- fix random breaks of the websocket under high load in the Torii
- fixed memory-corruption in segment-header of the clusters in Kyouko
- solved uninitialized warning in item-buffer
- fixed stupid memory-leak when sending internal messages

### Removed

- Removed additional namespaces of the libraries in the common-layer
- Disabled reduction-process of the neural-networkds for the moment. Will be re-added in 0.3.0



## [0.1.0] - 2022-10-18

### Added:

- First experimental prototype version with **very basic implementation** of following features:
    - first incomplete implementation of the core-concept of a dynamic network with:
        - creates connections while learning
        - has nearly no upper limit for inputs
    - capable to learn MNIST-Datasets and basic CSV-files
    - creating neuronal networks based on templates in json-format
    - authentication with username + password and JWT-token
    - creating multiple user and multiple projects in one system
    - basic role-system
    - automatic creation of REST-API-documentation in PDF-, RST- or Markdown-format
    - creating snapshots of neuronal networks and restore them again
    - thread-binding for all components for more optimal usage of the CPU
    - PoC to change CPU-frequency based on load
    - measuring of power consumption, thermal production and frequency of the CPU
    - Dashboard with HTML, Javasript and CSS with rendering on client-side
    - sdk-library in C++ and Javascript (incomplete)
    - Build-scripts to build docker-images for all components
    - basic Helm-chart to deploy all components on a kubernetes


### Repositories

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
Hanami-AI-Documentation | v0.1.0 | https://github.com/kitsudaiki/Hanami-AI-Documentation.git
Hanami-AI-Dockerbuilder | v0.1.0 | https://github.com/kitsudaiki/Hanami-AI-Dockerbuilder.git
Hanami-AI-Dashboard | v0.2.0 | https://github.com/kitsudaiki/Hanami-AI-Dashboard.git
Hanami-AI-Dashboard-Dependencies | v0.1.0 | https://github.com/kitsudaiki/anamiAI-Dashboard-Dependencies.git
Hanami-AI-K8s | v0.1.0 | https://github.com/kitsudaiki/Hanami-AI-K8s.git
libHanamiAiSdk | v0.4.0 | https://github.com/kitsudaiki/libHanamiAiSdk.git
KyoukoMind | v0.9.1 | https://github.com/kitsudaiki/KyoukoMind.git
MisakiGuard | v0.3.0 | https://github.com/kitsudaiki/MisakiGuard.git
ShioriArchive | v0.4.0 | https://github.com/kitsudaiki/ShioriArchive.git
AzukiHeart | v0.3.1 | https://github.com/kitsudaiki/AzukiHeart.git
ToriiGateway | v0.7.0 | https://github.com/kitsudaiki/ToriiGateway.git
TsugumiTester | v0.4.0 | https://github.com/kitsudaiki/TsugumiTester.git
libAzukiHeart | v0.3.0 | https://github.com/kitsudaiki/libAzukiHeart.git
libMisakiGuard | v0.2.0 | https://github.com/kitsudaiki/libMisakiGuard.git
ibShioriArchive | v0.3.0 | https://github.com/kitsudaiki/ibShioriArchive.git
libKitsunemimiHanamiMessages | v0.1.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiMessages.git
libKitsunemimiHanamiNetwork | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiNetwork.git
libKitsunemimiHanamiPolicies | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiPolicies.git
libKitsunemimiHanamiEndpoints | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiEndpoints.git
libKitsunemimiHanamiDatabase | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiDatabase.git
libKitsunemimiHanamiCommon | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git
libKitsunemimiSakuraNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraNetwork.git
libKitsunemimiSakuraLang | v0.13.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraLang.git
libKitsunemimiSakuraDatabase | v0.6.1 |  https://github.com/kitsudaiki/libKitsunemimiSakuraDatabase.git
libKitsunemimiSakuraHardware | v0.2.0 |  https://github.com/kitsudaiki/libKitsunemimiSakuraHardware.git
libKitsunemimiArgs | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | v0.5.0 |  https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiCrypto | v0.3.0 |  https://github.com/kitsudaiki/libKitsunemimiCrypto.git
libKitsunemimiCpu | v0.4.1 |  https://github.com/kitsudaiki/libKitsunemimiCpu.git
libKitsunemimiJwt | v0.5.1 |  https://github.com/kitsudaiki/libKitsunemimiJwt.git
libKitsunemimiSqlite | v0.4.0 |  https://github.com/kitsudaiki/libKitsunemimiSqlite.git
libKitsunemimiJson | v0.12.0 |  https://github.com/kitsudaiki/libKitsunemimiJson.git
libKitsunemimiJinja2 | v0.10.0 |  https://github.com/kitsudaiki/libKitsunemimiJinja2.git
libKitsunemimiIni | v0.6.0 |  https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiNetwork | v0.9.0 |  https://github.com/kitsudaiki/libKitsunemimiNetwork.git
libKitsunemimiCommon | v0.27.1 |  https://github.com/kitsudaiki/libKitsunemimiCommon.git
