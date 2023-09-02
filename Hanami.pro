TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++17

SUBDIRS =  src/libraries
SUBDIRS += src/sdk/cpp/hanami_sdk
SUBDIRS += src/Hanami
SUBDIRS += src/testing

src/sdk/cpp/hanami_sdk.depends = libraries
src/Hanami.depends = libraries hanami_sdk
src/testing.depends = core libraries hanami_sdk
