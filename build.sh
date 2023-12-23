#!/bin/bash

# get current directory-path and the path of the parent-directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR="$(dirname "$DIR")"

if [ $1 = "test" ]; then
    cmake -DCMAKE_BUILD_TYPE=Release -Drun_tests=ON  "$PARENT_DIR/Hanami"
else
    cmake -DCMAKE_BUILD_TYPE=Release "$PARENT_DIR/Hanami"
fi

/usr/bin/make -j8

#-----------------------------------------------------------------------------------------------------------------

