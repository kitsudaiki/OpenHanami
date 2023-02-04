#!/bin/bash

# get current directory-path and the path of the parent-directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR="$(dirname "$DIR")"

# create build-directory
BUILD_DIR="$PARENT_DIR/build"
mkdir -p $BUILD_DIR

# create directory for the final result
RESULT_DIR="$PARENT_DIR/result"
mkdir -p $RESULT_DIR

#-----------------------------------------------------------------------------------------------------------------

function build_kitsune_lib_repo () {
    REPO_NAME=$1
    NUMBER_OF_THREADS=$2
    ADDITIONAL_CONFIGS=$3

    # create build directory for repo and go into this directory
    REPO_DIR="$BUILD_DIR/$REPO_NAME"
    mkdir -p $REPO_DIR
    cd $REPO_DIR

    # build repo library with qmake
    /usr/lib/x86_64-linux-gnu/qt5/bin/qmake "$PARENT_DIR/$REPO_NAME/$REPO_NAME.pro" -spec linux-g++ "CONFIG += optimize_full $ADDITIONAL_CONFIGS"
    /usr/bin/make -j$NUMBER_OF_THREADS

    # copy build-result and include-files into the result-directory
    cp $REPO_DIR/src/$REPO_NAME.a $RESULT_DIR/
    cp -r $PARENT_DIR/$REPO_NAME/include $RESULT_DIR/
    ls -l $RESULT_DIR/include/
    ls -l $RESULT_DIR
}

function get_required_kitsune_lib_repo () {
    REPO_NAME=$1
    TAG_OR_BRANCH=$2
    NUMBER_OF_THREADS=$3
    ADDITIONAL_CONFIGS=$4

    # clone repo
    git clone  https://github.com/kitsudaiki/$REPO_NAME.git "$PARENT_DIR/$REPO_NAME"
    cd "$PARENT_DIR/$REPO_NAME"
    git checkout $TAG_OR_BRANCH

    build_kitsune_lib_repo $REPO_NAME $NUMBER_OF_THREADS $ADDITIONAL_CONFIGS
}

#-----------------------------------------------------------------------------------------------------------------

echo "###########################################################################################################"
echo ""
get_required_kitsune_lib_repo "libKitsunemimiCommon" "develop" 8 "staticlib"
get_required_kitsune_lib_repo "libKitsunemimiIni" "develop" 1 "staticlib"
get_required_kitsune_lib_repo "libKitsunemimiArgs" "develop" 8 "staticlib"
get_required_kitsune_lib_repo "libKitsunemimiConfig" "develop" 8 "staticlib"
echo ""
echo "###########################################################################################################"
echo ""
get_required_kitsune_lib_repo "libKitsunemimiHanamiCommon" "develop" 8 "staticlib"
echo ""
echo "###########################################################################################################"

#-----------------------------------------------------------------------------------------------------------------

if [ $1 = "test" ]; then
    build_kitsune_lib_repo "libKitsunemimiHanamiEndpoints" 1 "staticlib run_tests"
else
    build_kitsune_lib_repo "libKitsunemimiHanamiEndpoints" 1 "staticlib"
fi

#-----------------------------------------------------------------------------------------------------------------
