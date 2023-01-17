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

# create build directory for KyoukoMind and go into this directory
LIB_HANAMI_DIR="$BUILD_DIR/Hanami-AI"
mkdir -p $LIB_HANAMI_DIR
cd $LIB_KITSUNE_HANAMI_DIR

# build KyoukoMind library with qmake
/usr/lib/x86_64-linux-gnu/qt5/bin/qmake "$PARENT_DIR/Hanami-AI/Hanami-AI.pro" -spec linux-g++ "CONFIG += optimize_full"
/usr/bin/make -j1
# copy build-result and include-files into the result-directory
# cp "$LIB_HANAMI_DIR/KyoukoMind" "$RESULT_DIR/"

#-----------------------------------------------------------------------------------------------------------------

