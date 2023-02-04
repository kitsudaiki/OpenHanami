#!/bin/bash

# get current directory-path and the path of the parent-directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR="$(dirname "$DIR")"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# create build-directory
BUILD_DIR="$PARENT_DIR/build"
mkdir -p $BUILD_DIR

# create directory for the final result
RESULT_DIR="$PARENT_DIR/result"
mkdir -p $RESULT_DIR

#-----------------------------------------------------------------------------------------------------------------

# create build directory for Hanami-AI and go into this directory
LIB_HANAMI_DIR="$BUILD_DIR/Hanami-AI"
mkdir -p $LIB_HANAMI_DIR
cd $LIB_KITSUNE_HANAMI_DIR

# build Hanami-AI library with qmake
/usr/lib/x86_64-linux-gnu/qt5/bin/qmake "$PARENT_DIR/Hanami-AI/Hanami-AI.pro" -spec linux-g++ "CONFIG += optimize_full"

# IMPORTNANT: at the moment it has to be build with only 1 thread, because the parser-generation with bison and flex
#             has problems in a parallel build-process (see issue #30)
/usr/bin/make -j1

#-----------------------------------------------------------------------------------------------------------------

