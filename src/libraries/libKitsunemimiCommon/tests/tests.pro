TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++17

SUBDIRS = \
    unit_tests \
    memory_leak_tests

tests.depends = src
