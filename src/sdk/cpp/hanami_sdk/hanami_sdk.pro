TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++14

SUBDIRS = src

run_tests {
    SUBDIRS += tests

    tests.depends = src
}

