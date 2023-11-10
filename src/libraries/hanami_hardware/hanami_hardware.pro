TEMPLATE = subdirs
CONFIG += ordered

QMAKE_CXX = clang++-15
QMAKE_LINK = clang++-15

SUBDIRS = src

run_tests {
    SUBDIRS += tests

    tests.depends = src
}

