# hanami_args

!!! warning

    This documentation here is the archived version of the old readme-file of this library and is NOT up-to-date, but maybe it is still useful for some references.

## Description

Small and easy to use parser for CLI-arguments.

## Usage by example

This example should show, how the parser is used and what is possible.

HINT: The flags `--help` and `-h` for the help-output are hard coded and don't have to be set.

```cpp
#include <hanami_args/arg_parser.h>
#include <hanami_common/logger/logger.h>
#include <hanami_common/logger.h>

int main(int argc, char *argv[])
{
    // error messages of the parser are printed via logger
    Hanami::initConsoleLogger(true);
    // with "initFileLogger" the error-message of the argument-parser can also be written into a file

    Hanami::ArgParser parser("0.1.0");
    Hanami::ErrorContainer error;

    // register flags without value
    parser.registerPlain("debug,d",
                         "debug-flag to enable addtional debug output",
                         error);
    // "registerPlain" allows it to register flags without any value, which says only true or flase
    //                 if they were set or not set

    // register flags
    parser.registerString("source",
                          "source-path",
                          error,
                          true);
    parser.registerInteger("input,i",
                           "additional parameter");

    // register other values
    parser.registerString("mode",
                          "modus for converting",
                          error,
                          true,  // true to make it requried
                          true); // true to register this without a "--"-flag
    parser.registerString("destination",
                          "destination path for output",
                          error,
                          true,
                          true);
    // register types:
    //     registerString
    //     registerInteger
    //     registerFloat
    //     registerBoolean

    // parse incoming arguments
    bool ret = parser.parse(argc, argv, error);
    if(ret == false)
    {
        LOG_ERROR(error);
        return 1;
    }
    // ret say, if the converting was successful or not. Error-message are written in the logger

    // check if flags without values were set. In this case check if the debug-flag was set
    bool debug = parser.wasSet("debug");

    // get values with or without flag as list of value for the case, that a flag was
    // used multiple times within one cli-call:
    const std::vector<std::string> testValues = parser.getStringValues("source");
    const std::vector<long> numbers = parser.getIntValues("input");
    // get types:
    //     getStringValues
    //     getIntValues
    //     getFloatValues
    //     getBoolValues

    // get values without flag:
    const std::string mode = parser.getStringValue("mode");
    const std::string destination = parser.getStringValue("destination");
    // get types:
    //     getStringValue
    //     getIntValue
    //     getFloatValue
    //     getBoolValue

    //...

    return 0;
}

```

If the tool would called `cli_test` the command `cli_test --help` would produce the following
output:

```
command: cli_test [options] --source ... <mode> <destination>

Options:
+-----------+-------+--------+-------------+---------------------------------------------+
| long      | short | type   | is required | help-text                                   |
+===========+=======+========+=============+=============================================+
| --help    | -h    |        |             | print help ouput                            |
+-----------+-------+--------+-------------+---------------------------------------------+
| --version | -v    |        |             | print program version                       |
+-----------+-------+--------+-------------+---------------------------------------------+
| --debug   | -d    |        |             | debug-flag to enable addtional debug output |
+-----------+-------+--------+-------------+---------------------------------------------+
| --source  |       | string | x           | source-path                                 |
+-----------+-------+--------+-------------+---------------------------------------------+
| --input   | -i    | number |             | additional parameter                        |
+-----------+-------+--------+-------------+---------------------------------------------+

Required:
+---------------+--------+-----------------------------+
| name          | type   | text                        |
+===============+========+=============================+
| <mode>        | string | modus for converting        |
+---------------+--------+-----------------------------+
| <destination> | number | destination path for output |
+---------------+--------+-----------------------------+
```

If this example is called with a string `asdf` for the flag `-i`, the error-message looks like this:

```
ERROR: argument has the false type:
    required type: number
    identifier: -i
    given value: asdf
```
