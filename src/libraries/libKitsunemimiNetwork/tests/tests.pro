TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++14

SUBDIRS = \
    functional_tests \
    memory_leak_tests

tests.depends = src
