/**
 *  @file       arg_parser.h
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  Apache License Version 2.0
 *
 *      Copyright 2022 Tobias Anker
 *
 *      Licensed under the Apache License, Version 2.0 (the "License");
 *      you may not use this file except in compliance with the License.
 *      You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 *      Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *      See the License for the specific language governing permissions and
 *      limitations under the License.
 */

#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <iostream>
#include <vector>
#include <climits>
#include <cstdlib>
#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{
class DataItem;
class DataArray;
class ArgParser_Test;
class SubCommand;

class ArgParser
{
public:
    ArgParser(const std::string &version = "");
    ~ArgParser();

    // register
    bool registerPlain(const std::string &identifier,
                       const std::string &helpText,
                       ErrorContainer &error);
    bool registerString(const std::string &identifier,
                        const std::string &helpText,
                        ErrorContainer &error,
                        bool required = false,
                        bool withoutFlag = false);
    bool registerInteger(const std::string &identifier,
                         const std::string &helpText,
                         ErrorContainer &error,
                         bool required = false,
                         bool withoutFlag = false);
    bool registerFloat(const std::string &identifier,
                       const std::string &helpText,
                       ErrorContainer &error,
                       bool required = false,
                       bool withoutFlag = false);
    bool registerBoolean(const std::string &identifier,
                         const std::string &helpTex,
                         ErrorContainer &errort,
                         bool required = false,
                         bool withoutFlag = false);

    // parse
    bool parse(const int argc,
               char *argv[],
               ErrorContainer &error);
    bool parse(const int argc,
               const char* argv[],
               ErrorContainer &error);

    // getter
    uint64_t getNumberOfValues(const std::string &identifier);
    bool wasSet(const std::string &identifier);
    const std::vector<std::string> getStringValues(const std::string &identifier);
    const std::vector<long> getIntValues(const std::string &identifier);
    const std::vector<double> getFloatValues(const std::string &identifier);
    const std::vector<bool> getBoolValues(const std::string &identifier);

    const std::string getStringValue(const std::string &identifier);
    long getIntValue(const std::string &identifier);
    double getFloatValue(const std::string &identifier);
    bool getBoolValue(const std::string &identifier);

private:
    friend ArgParser_Test;
    friend SubCommand;

    enum ArgType
    {
        NO_TYPE,
        STRING_TYPE,
        INT_TYPE,
        FLOAT_TYPE,
        BOOL_TYPE
    };

    struct ArgDefinition
    {
        bool withoutFlag = false;
        bool required = false;
        bool hasValue = false;
        bool wasSet = false;
        std::string longIdentifier = "";
        std::string shortIdentifier = "";
        ArgType type = STRING_TYPE;
        std::string helpText = "";

        DataArray* results = nullptr;
    };

    uint32_t m_positionCounter = 0;
    std::string m_version = "";
    std::vector<ArgDefinition> m_argumentList;

    const std::string convertType(ArgType type);
    void print(const std::string &commandName);
    bool precheckFlags(const int argc, const char* argv[]);

    ArgDefinition* getArgument(const std::string &identifier);
    bool registerArgument(const std::string &identifier,
                          const std::string &helpText,
                          const ArgType type,
                          bool required,
                          bool withoutFlag,
                          bool hasValue,
                          ErrorContainer &error);

    DataItem* convertValue(const std::string &value,
                           const ArgType requiredType);
};

}

#endif // ARG_PARSER_H
