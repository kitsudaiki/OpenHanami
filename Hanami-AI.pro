TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++17

SUBDIRS =  src/libraries
SUBDIRS += src/components
SUBDIRS += src/sdk/cpp/libHanamiAiSdk

src/sdk/cpp/libHanamiAiSdk.depends = libraries
src/components.depends = libraries libHanamiAiSdk
