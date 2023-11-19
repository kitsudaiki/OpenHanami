TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++17

QMAKE_CXX = clang++-15
QMAKE_LINK = clang++-15

SUBDIRS =  src/libraries
SUBDIRS += src/Hanami

src/Hanami.depends = libraries
src/testing.depends = core libraries
