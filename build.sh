#!/bin/bash

# get current directory-path and the path of the parent-directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR="$(dirname "$DIR")"

# build Hanami-AI library with qmake
/usr/lib/x86_64-linux-gnu/qt5/bin/qmake "$PARENT_DIR/Hanami-AI/Hanami-AI.pro" -spec linux-g++ "CONFIG += optimize_full staticlib"

# IMPORTNANT: at the moment it has to be build with only 1 thread, because the parser-generation with bison and flex
#             has problems in a parallel build-process (see issue #30)
/usr/bin/make -j1

#-----------------------------------------------------------------------------------------------------------------

