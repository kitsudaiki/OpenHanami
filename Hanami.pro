TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++17

QMAKE_CXX = clang++-15
QMAKE_LINK = clang++-15

SUBDIRS =  src/libraries
SUBDIRS += src/sdk/cpp/hanami_sdk
SUBDIRS += src/Hanami

src/sdk/cpp/hanami_sdk.depends = libraries
src/Hanami.depends = libraries hanami_sdk
src/testing.depends = core libraries hanami_sdk
