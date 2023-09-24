QT -= qt core gui

TARGET = hanami_policies
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.1.0

LIBS += -L../../hanami_args/src -lhanami_args
LIBS += -L../../hanami_args/src/debug -lhanami_args
LIBS += -L../../hanami_args/src/release -lhanami_args
INCLUDEPATH += ../../hanami_args/include

LIBS += -L../../hanami_ini/src -lhanami_ini
LIBS += -L../../hanami_ini/src/debug -lhanami_ini
LIBS += -L../../hanami_ini/src/release -lhanami_ini
INCLUDEPATH += ../../hanami_ini/include

LIBS += -L../../hanami_config/src -lhanami_config
LIBS += -L../../hanami_config/src/debug -lhanami_config
LIBS += -L../../hanami_config/src/release -lhanami_config
INCLUDEPATH += ../../hanami_config/include

LIBS += -L../../hanami_common/src -lhanami_common
LIBS += -L../../hanami_common/src/debug -lhanami_common
LIBS += -L../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../hanami_common/include

INCLUDEPATH += ../../../third-party-libs/json/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

SOURCES += \
    policy.cpp \
    policy_parsing/policy_parser_interface.cpp

HEADERS += \
    ../include/hanami_policies/policy.h \
    policy_parsing/policy_parser_interface.h


FLEXSOURCES = grammar/policy_lexer.l
BISONSOURCES = grammar/policy_parser.y

OTHER_FILES +=  \
    $$FLEXSOURCES \
    $$BISONSOURCES

# The following code calls the flex-lexer and bison-parser before compiling the
# cpp-code for automatic generation of the parser-code in each build-step.
# The resulting source-code-files are stored in the build-directory of the policy-converter.
flexsource.input = FLEXSOURCES
flexsource.output = ${QMAKE_FILE_BASE}.cpp
flexsource.commands = flex --header-file=${QMAKE_FILE_BASE}.h -o ${QMAKE_FILE_BASE}.cpp ${QMAKE_FILE_IN}
flexsource.variable_out = SOURCES
flexsource.name = Flex Sources ${QMAKE_FILE_IN}
flexsource.CONFIG += target_predeps
flexsource.CONFIG += target_predeps

QMAKE_EXTRA_COMPILERS += flexsource

flexheader.input = FLEXSOURCES
flexheader.output = ${QMAKE_FILE_BASE}.h
flexheader.commands = @true
flexheader.variable_out = HEADERS
flexheader.name = Flex Headers ${QMAKE_FILE_IN}
flexheader.CONFIG += target_predeps
flexheader.CONFIG += target_predeps no_link

QMAKE_EXTRA_COMPILERS += flexheader

bisonsource.input = BISONSOURCES
bisonsource.output = ${QMAKE_FILE_BASE}.cpp
bisonsource.commands = bison -d --defines=${QMAKE_FILE_BASE}.h -o ${QMAKE_FILE_BASE}.cpp ${QMAKE_FILE_IN}
bisonsource.variable_out = SOURCES
bisonsource.name = Bison Sources ${QMAKE_FILE_IN}
bisonsource.CONFIG += target_predeps
bisonsource.CONFIG += target_predeps

QMAKE_EXTRA_COMPILERS += bisonsource

bisonheader.input = BISONSOURCES
bisonheader.output = ${QMAKE_FILE_BASE}.h
bisonheader.commands = @true
bisonheader.variable_out = HEADERS
bisonheader.name = Bison Headers ${QMAKE_FILE_IN}
bisonheader.CONFIG += target_predeps
bisonheader.CONFIG += target_predeps no_link

QMAKE_EXTRA_COMPILERS += bisonheader
