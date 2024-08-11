# Dependencies

## Backend

### Build hanami

| apt-package        | Purpose                                                                              |
| ------------------ | ------------------------------------------------------------------------------------ |
| clang-15           | C++-compiler                                                                         |
| clang-format-15    | Helper for styling the source-code                                                   |
| make               | Build-Tool                                                                           |
| cmake              | Tool to genereate make-file for the code.                                            |
| bison              | Parser-Generator                                                                     |
| flex               | Lexxer for the Parser-Generator                                                      |
| libssl-dev         | ssl-library for TCS-encryption of network-connections                                |
| libcrypto++-dev    | HMAC, SHA256 and other crypto related operations                                     |
| libboost1.74-dev   | Provides the Beast-library of Boost, which is used for the REST-API within the Torii |
| uuid-dev           | Generate UUID's within the code                                                      |
| libsqlite3-dev     | Library to interact with the SQLite3 databases                                       |
| protobuf-compiler  | Convert protobuf-files into source-code                                              |
| nlohmann-json3-dev | Json-parser                                                                          |

### Submodules

| name               | License | Purpose                       |
| ------------------ | ------- | ----------------------------- |
| Thalhammer/jwt-cpp | MIT     | create and validate jwt-token |

### Runtime

| apt-package   | Purpose                                                                           |
| ------------- | --------------------------------------------------------------------------------- |
| openssl       | ssl-library for TCS-encryption of network-connections                             |
| libuuid1      | Generate UUID's within the code                                                   |
| libcrypto++8  | HMAC, SHA256 and other crypto related operations                                  |
| libsqlite3-0  | Library to interact with the SQLite3 databases                                    |
| libprotobuf23 | Runtime-library for protobuffers                                                  |
| libboost1.74  | Provides the Beast-library of Boost, which is used for the REST-API within Hanami |

### Supported compiler

| C++ Compiler                                                  |
| ------------------------------------------------------------- |
| [![ubuntu-2204_clang-13][img_ubuntu-2204_clang-13]][Workflow] |
| [![ubuntu-2204_clang-14][img_ubuntu-2204_clang-14]][Workflow] |
| [![ubuntu-2204_clang-15][img_ubuntu-2204_clang-15]][Workflow] |
| [![ubuntu-2404_clang-15][img_ubuntu-2404_clang-15]][Workflow] |
| [![ubuntu-2404_clang-16][img_ubuntu-2404_clang-16]][Workflow] |
| [![ubuntu-2404_clang-17][img_ubuntu-2404_clang-17]][Workflow] |
| [![ubuntu-2404_clang-18][img_ubuntu-2404_clang-18]][Workflow] |
| [![ubuntu-2204_gcc-10][img_ubuntu-2204_gcc-10]][Workflow]     |
| [![ubuntu-2204_gcc-11][img_ubuntu-2204_gcc-11]][Workflow]     |
| [![ubuntu-2204_gcc-12][img_ubuntu-2204_gcc-12]][Workflow]     |
| [![ubuntu-2404_gcc-12][img_ubuntu-2404_gcc-12]][Workflow]     |
| [![ubuntu-2404_gcc-13][img_ubuntu-2404_gcc-13]][Workflow]     |
| [![ubuntu-2404_gcc-14][img_ubuntu-2404_gcc-14]][Workflow]     |

## Python-SDK

### Packages

see
[requirements.txt](https://github.com/kitsudaiki/Hanami/blob/develop/src/sdk/python/hanami_sdk/requirements.txt)

### Suppored Python-versions

| Python (SDK)                                |
| ------------------------------------------- |
| [![python-3_9][img_python-3_9]][Workflow]   |
| [![python-3_10][img_python-3_10]][Workflow] |
| [![python-3_11][img_python-3_11]][Workflow] |
| [![python-3_12][img_python-3_12]][Workflow] |

## Go CLI-client

see [go.sum](https://github.com/kitsudaiki/Hanami/blob/develop/src/cli/hanamictl/go.sum)

[img_ubuntu-2204_clang-13]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2204_clang-13/shields.json&style=flat-square
[img_ubuntu-2204_clang-14]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2204_clang-14/shields.json&style=flat-square
[img_ubuntu-2204_clang-15]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2204_clang-15/shields.json&style=flat-square
[img_ubuntu-2404_clang-15]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2404_clang-15/shields.json&style=flat-square
[img_ubuntu-2404_clang-16]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2404_clang-16/shields.json&style=flat-square
[img_ubuntu-2404_clang-17]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2404_clang-17/shields.json&style=flat-square
[img_ubuntu-2404_clang-18]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2404_clang-18/shields.json&style=flat-square
[img_ubuntu-2204_gcc-10]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2204_gcc-10/shields.json&style=flat-square
[img_ubuntu-2204_gcc-11]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2204_gcc-11/shields.json&style=flat-square
[img_ubuntu-2204_gcc-12]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2204_gcc-12/shields.json&style=flat-square
[img_ubuntu-2404_gcc-12]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2404_gcc-12/shields.json&style=flat-square
[img_ubuntu-2404_gcc-13]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2404_gcc-13/shields.json&style=flat-square
[img_ubuntu-2404_gcc-14]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/compiler_version/ubuntu-2404_gcc-14/shields.json&style=flat-square
[img_python-3_9]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/python_version/python-3_9/shields.json&style=flat-square
[img_python-3_10]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/python_version/python-3_10/shields.json&style=flat-square
[img_python-3_11]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/python_version/python-3_11/shields.json&style=flat-square
[img_python-3_12]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/Hanami-badges/develop/python_version/python-3_12/shields.json&style=flat-square
[Workflow]: https://github.com/kitsudaiki/Hanami/actions/workflows/build_test.yml
