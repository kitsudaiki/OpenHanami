#!/bin/bash

# get current directory-path and the path of the parent-directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR="$(dirname "$DIR")"

if [ $1 = "test" ]; then
    /usr/lib/x86_64-linux-gnu/qt5/bin/qmake "$PARENT_DIR/Hanami/Hanami.pro" -spec linux-g++ "CONFIG += optimize_full staticlib run_tests"
else
    /usr/lib/x86_64-linux-gnu/qt5/bin/qmake "$PARENT_DIR/Hanami/Hanami.pro" -spec linux-g++ "CONFIG += optimize_full staticlib"
fi

/usr/bin/make -j8

#-----------------------------------------------------------------------------------------------------------------

