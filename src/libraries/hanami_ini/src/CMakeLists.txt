cmake_minimum_required(VERSION 3.15)

project(hanami_ini VERSION 0.5.1)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_COMPILER "clang++-15")

add_library(hanami_ini SHARED)


# Flex and Bison commands
find_program(FLEX_EXECUTABLE NAMES flex)
find_program(BISON_EXECUTABLE NAMES bison)

if(FLEX_EXECUTABLE AND BISON_EXECUTABLE)
    set(FLEX_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/grammar/ini_lexer.l)
    set(FLEX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.cpp)

    set(BISON_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/grammar/ini_parser.y)
    set(BISON_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.cpp)

    add_custom_command(
        OUTPUT ${FLEX_OUTPUT} ${BISON_OUTPUT}
        COMMAND ${FLEX_EXECUTABLE} --header-file=${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.h -o ${FLEX_OUTPUT} ${FLEX_INPUT}
        COMMAND ${BISON_EXECUTABLE} -d --defines=${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.h -o ${BISON_OUTPUT} ${BISON_INPUT}
        DEPENDS ${FLEX_INPUT} ${BISON_INPUT}
    )

    target_sources(hanami_ini PUBLIC ${FLEX_OUTPUT} ${BISON_OUTPUT})
endif()




