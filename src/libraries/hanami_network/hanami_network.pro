TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++17

SUBDIRS = src

run_tests {
    SUBDIRS += tests

    tests.depends = src
}

