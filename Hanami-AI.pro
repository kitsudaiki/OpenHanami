TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++17

SUBDIRS =  src/libraries
SUBDIRS += src/sdk/cpp/libHanamiAiSdk
SUBDIRS += src/Hanami
SUBDIRS += src/testing

src/sdk/cpp/libHanamiAiSdk.depends = libraries
src/Hanami.depends = libraries libHanamiAiSdk
src/testing.depends = core libraries libHanamiAiSdk
