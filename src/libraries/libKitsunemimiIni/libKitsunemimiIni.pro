TEMPLATE = subdirs
CONFIG += ordered

lexxer.file = src/lexxer.pro
parser.file = src/parser.pro

SUBDIRS = parser
SUBDIRS += lexxer
SUBDIRS += src

lexxer.depends = parser
src.depends = lexxer

run_tests {
    SUBDIRS += tests

    tests.depends = src
}

