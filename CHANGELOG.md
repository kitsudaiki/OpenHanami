# Changelog

## v0.4.0

- **Date**: -

### Breaking-Changes

#### Checkpoint-Breaking

- Complete restructure of the core to improve the performance of the CUDA-kernel by enabling more capabilities for parallel processing.
- Synapse-blocks are not limited to a single cluster-scoped buffer, but instead are stored in a global buffer, which is shared by multiple cluster.
- The number of used synapse-blocks is now independed from the number of neuron-blocks of a brick. They scale now better over time. This way the number of neurons in the clsuter-template are only a maximum value. This way it is easier to too high values in the cluster-template are no pain-point anymore.

#### API-Breaking

- removed c++ version of the SDK

### Added

- multi-threading, so now multiple cpu-threads can process the same cluster at the same time
- new endpoint to list logical hosts (cpu's and gpu's)
- new endpoint to move a cluster between cpu and gpu
- re-added the spiking neural network-feature for future testing-purpose
- re-added a connection-distance of more the one for future testing-purpose
- python version of the SDK
- basic cli-client in go 
- new entry in config to define a location for temporary files while uploading
- new cleanup-loop for tempfiles, which deleted incative temporary files after a certain amount of time, to remove file of a broken upload-process
- the reduction-process was re-added to limit the amount of used memory while learning.

### Changed

- physical host is now separated into logical hosts (a cpu and a gpu are handled as 2 different logical hosts)
- replaced old api-test-tool by a new python-script, which use and test the new python version of the SDK
- moved from `qmake` to `cmake` as build-tool for the c++ code

### Removed

- removed GET-requests from audit-log
- removed c++ version of the SDK



## v0.3.3

- **Date**: 2023-11-17

### Fixed

- fixed broken login to dashboard
- fixed broken internal error-log and the error-log API-endpoint



## v0.3.2

- **Date**: 2023-11-10

### Added

- basic Ansible-playbook was added in order to deploy the project without kubernetes
- in kubernetes-setup:
    - now use cert-manager to generate certificates 
    - ingress was added for ssl-termination of https-connections
    - persistent volume was added to persist data like database within the kuberntes-setup
    - node label was added to define the node, where hanami should be deployed by kubernetes
- `clang-format` file was added with a git pre-commit hook
- `clang-format` check and basic `cppcheck` were added to the ci-pipeline
- first contribution-guide and code-styling guide was added
- example-configs for testing purpose and guide to setup local test-environment
- at start there is now a check if the directories for the checkpoints and dataset, defined by the config, even exist

### Fixed

- fixed compile-error when trying to build on ARM64 architecture
- fixed compile-error in cude-code with the standard nvidia-toolkit in ubuntu 22.04
- fixed broken json-strings in javascript-sdk
- fixed false error response-codes in API

### Changed

- moved old readme files into the normal documentation
- use clang++ instead of g++ as compiler
- non-critical API-errors, like for example a 404 (not found) when searching for an unknown id, doesn't produce an internal error-output anymore 

### Removed

- ssl-termination was removed from backend-API, because it will be done by another service in front of the API, like nginx



## v0.3.1

- **Date**: 2023-09-24

### Changed

- replaced json-parser by third-party library https://github.com/nlohmann/json
- updated internal registration of config-values
- updated internal registration of cli-arguments

### Fixed

- fixed bug in synapse-segmentation, which had impact in samller tests

### Removed

- Removed unused and deprecated checkpoint create and finalize endpoints



## v0.3.0 

- **Date**: 2023-09-05

### Breaking-Changes

#### All breaking

- Move the entire project from a micro-service architecture to a more monolithic architecture with only one executable but still multiple libraries, to reduce unnecessary complexity and increase performance and reliability of the program. Also it makes the developing process easier and faster, with is necessary in regard of my limited time-resources.
- Removed entire segment-layer between the clusters and bricks from core structure. This layer was originally intended for the separation of multiple threads, but now there is another strategy planed 

#### API-Breaking

- REST-API endpoint for the generation of the documentation was removed
- Renamed `snapshot` to `checkpoint` and `learn` to `train`

#### Database-Breaking

- fixed sizes of some database-columns for user- and project-id's

#### Checkpoint-Breaking

- Bigger rework of the core structs to get rid of the update-positions struct

### Added

- documentation of config-entries and database-schemas can now also be generated as marddown-documents
- error-codes are now also written into the REST-API documentation
- new config-entry to enable the extremely experimental CUDA-processing
- more then one thread is now created for the core-processing, to be able to process multiple clusters in parallel
- the CUDA-version now also works with Checkpoints (create and read)

### Changed

- Rename repository/project from `Hanami-AI` to `Hanami`
- Renamed libraries to `hanami_...` and namespaces of the libraries to `Hanami`, because the originally naming and structure was from the time, when they were separate repositores and desired also for other projects. The new naming makes the names shorter and easier.
- replace custom jwt lib by `jwt-cpp` (https://github.com/Thalhammer/jwt-cpp)
- REST-API documentation is now generated as OpenAPI-specification and added to the documentation via swagger-ui
- merged networking-libraries in context of the renaming
- config-handler was internally restructured for the new generator of the config-documentation
- registration of fields in API-endpoints was updated for cleaner code
- Button in dashboard for the geneartion of the Documentation was removed and is now done by a new CLI flag.
- merged entire code into one single docker-image
- better segmentation of the synapses

### Fixed

- solved all compiler-warnings (disabled libraries not included)
- fixed stupid memory leaks in API and task-handling
- fixed handling in database-requests to separate correctly between an internal-error and a not-found to give the correct HTTP response to the user
- after a restart of the backend, all clusters are not removed from database at the start to avoid broken clusters, because cluster are in-memory and don't survive a restart
- positioning of header-texts in dashboard was fixed

### Removed

- Removed OpenCL-kernel for the moment. The CUDA-variant is enough for testing and easier to update.



## v0.2.0 

- **Date**: 2023-03-15

### Breaking-Changes

#### API-Breaking

- use now custom syntax for cluster- and segment-templates instead of json
- use protobuf-messages to send payload of files
- merge endpoints to create train- and request-tasks for csv- and mnist-files into one endpoint
- creating a task doesn't require the explicit selecting of the data-set type. This is now read from the data-set metadata

#### Checkpoint-Breaking

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
- Disabled reduction-process of the neural-networkds for the moment. 



## v0.1.0

- **Date**: 2022-10-18

### Added

- First experimental prototype version with **very basic implementation** of following features:
    - first incomplete implementation of the core-concept of a dynamic network with:
        - creates connections while learning
        - has nearly no upper limit for inputs
    - capable to train MNIST-Datasets and basic CSV-files
    - creating neuronal networks based on templates in json-format
    - authentication with username + password and JWT-token
    - creating multiple user and multiple projects in one system
    - basic role-system
    - automatic creation of REST-API-documentation in PDF-, RST- or Markdown-format
    - creating checkpoints of neuronal networks and restore them again
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
