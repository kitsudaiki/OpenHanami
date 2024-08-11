cmake_minimum_required(VERSION 3.16)
project(hanami_policies VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(hanami_policies STATIC
    include/hanami_policies/policy.h
    src/policy.cpp
    src/policy_parsing/policy_parser_interface.h
    src/policy_parsing/policy_parser_interface.cpp
    src/grammar/policy_lexer.l
    src/grammar/policy_parser.y
)

target_include_directories(hanami_policies PUBLIC
    include/
    src/
    ../hanami_common/include
)

target_link_libraries(hanami_policies PUBLIC
    hanami_common
)

# Flex and Bison commands
find_program(FLEX_EXECUTABLE NAMES flex)
find_program(BISON_EXECUTABLE NAMES bison)

if(FLEX_EXECUTABLE AND BISON_EXECUTABLE)
    set(FLEX_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/grammar/policy_lexer.l)
    set(FLEX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/policy_lexer.cpp)

    set(BISON_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/grammar/policy_parser.y)
    set(BISON_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/policy_parser.cpp)

    add_custom_command(
        OUTPUT ${FLEX_OUTPUT} ${BISON_OUTPUT}
        COMMAND ${BISON_EXECUTABLE} -d --defines=${CMAKE_CURRENT_BINARY_DIR}/policy_parser.h -o ${BISON_OUTPUT} ${BISON_INPUT}
        COMMAND ${FLEX_EXECUTABLE} --header-file=${CMAKE_CURRENT_BINARY_DIR}/policy_lexer.h -o ${FLEX_OUTPUT} ${FLEX_INPUT}
        DEPENDS ${FLEX_INPUT} ${BISON_INPUT}
    )

    # Add dependencies on the targets representing the generated files
    add_custom_target(generate_policy_flex_bison_files DEPENDS ${FLEX_OUTPUT} ${BISON_OUTPUT})

    # Make hanami_policies depend on the custom target
    add_dependencies(hanami_policies generate_policy_flex_bison_files)

    target_sources(hanami_policies PRIVATE ${FLEX_OUTPUT} ${BISON_OUTPUT})

endif()

if(run_tests)
    add_subdirectory(tests)
endif()
