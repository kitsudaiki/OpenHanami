TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++17

QMAKE_CXX = clang++-15
QMAKE_LINK = clang++-15

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

