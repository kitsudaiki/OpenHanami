#!/bin/bash

# get current directory-path and the path of the parent-directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR="$(dirname "$DIR")"

if [ $1 = "test" ]; then
    QMAKE_CXX=clang++  /usr/lib/x86_64-linux-gnu/qt5/bin/qmake "$PARENT_DIR/Hanami/Hanami.pro" -r -spec linux-clang "CONFIG += optimize_full staticlib run_tests"
else
    QMAKE_CXX=clang++  /usr/lib/x86_64-linux-gnu/qt5/bin/qmake "$PARENT_DIR/Hanami/Hanami.pro" -r -spec linux-clang "CONFIG += optimize_full staticlib"
fi

/usr/bin/make -j8

#-----------------------------------------------------------------------------------------------------------------

