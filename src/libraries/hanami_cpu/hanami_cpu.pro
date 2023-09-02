TEMPLATE = subdirs
CONFIG += ordered

SUBDIRS = src

run_tests {
    SUBDIRS += tests

    tests.depends = src
}

