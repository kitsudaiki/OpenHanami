# libKitsunemimiHanamiEndpoints (DEPRECATED)

![Github workfloat status](https://img.shields.io/github/workflow/status/kitsudaiki/libKitsunemimiHanamiEndpoints/build-and-test/develop?label=build%20and%20test&style=flat-square)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/kitsudaiki/libKitsunemimiHanamiEndpoints?label=version&style=flat-square)
![GitHub](https://img.shields.io/github/license/kitsudaiki/libKitsunemimiHanamiEndpoints?style=flat-square)
![C++Version](https://img.shields.io/badge/c%2B%2B-17-blue?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Linux--x64-lightgrey?style=flat-square)

## Description

**IMPORTANT: This library is no longer maintained** 

Handling and checking of custom endpoint-files.

## Build

### Requirements

name | repository | version | task
--- | --- | --- | ---
g++ | g++ | >= 8.0 | Compiler for the C++ code.
make | make | >= 4.0 | process the make-file, which is created by qmake to build the programm with g++
qmake | qt5-qmake | >= 5.0 | This package provides the tool qmake, which is similar to cmake and create the make-file for compilation.
FLEX | flex | >= 2.6 | Build the lexer-code for all used parser.
GNU Bison | bison | 3.x | Build the parser-code together with the lexer-code

Installation on Ubuntu/Debian:

```bash
sudo apt-get install g++ make qt5-qmake bison flex
```

IMPORTANT: All my projects are only tested on Linux.

### Kitsunemimi-repositories

Repository-Name | Version-Tag | Download-Path
--- | --- | ---
libKitsunemimiCommon | develop | https://github.com/kitsudaiki/libKitsunemimiCommon.git
libKitsunemimiIni | develop | https://github.com/kitsudaiki/libKitsunemimiIni.git
libKitsunemimiArgs | develop | https://github.com/kitsudaiki/libKitsunemimiArgs.git
libKitsunemimiConfig | develop | https://github.com/kitsudaiki/libKitsunemimiConfig.git
libKitsunemimiHanamiCommon | develop | https://github.com/kitsudaiki/libKitsunemimiHanamiCommon.git

HINT: These Kitsunemimi-Libraries will be downloaded and build automatically with the build-script below.

### build library

In all of my repositories you will find a `build.sh`. You only have to run this script. It doesn't required sudo, because you have to install required tool via apt, for example, by yourself. But if other projects from me are required, it download them from github and build them in the correct version too. This script is also use by the ci-pipeline, so its tested with every commit.


Run the following commands:

```
git clone https://github.com/kitsudaiki/libKitsunemimiHanamiEndpoints.git
cd libKitsunemimiHanamiEndpoints
./build.sh
cd ../result
```

It create automatic a `build` and `result` directory in the directory, where you have cloned the project. At first it build all into the `build`-directory and after all build-steps are finished, it copy the include directory from the cloned repository and the build library into the `result`-directory. So you have all in one single place.

Tested on Debian and Ubuntu. If you use Centos, Arch, etc and the build-script fails on your machine, then please write me a mail and I will try to fix the script.

## Usage

(sorry, docu comes later)


## Contributing

Please give me as many inputs as possible: Bugs, bad code style, bad documentation and so on.

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](LICENSE) file for details
