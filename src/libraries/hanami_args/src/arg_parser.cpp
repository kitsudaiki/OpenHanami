/**
 *  @file       arg_parser.cpp
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

#include <hanami_args/arg_parser.h>

#include <hanami_common/methods/string_methods.h>
#include <hanami_common/items/data_items.h>
#include <hanami_common/items/table_item.h>

#include <hanami_common/logger.h>

namespace Kitsunemimi
{

/**
 * @brief constructor
 */
ArgParser::ArgParser(const std::string &version)
{
    ErrorContainer error;
    // register the help-output as special case
    registerPlain("help,h", "print help ouput", error);

    if(version != "")
    {
        m_version = version;
        registerPlain("version,v", "print program version", error);
    }
}

/**
 * @brief destructor
 */
ArgParser::~ArgParser()
{
    for(uint32_t i = 0; i < m_argumentList.size(); i++)
    {
        delete m_argumentList[i].results;
    }
}

/**
 * @brief register argument without value
 *
 * @param identifier Identifier for the new argument. Its a single word like "flag" for defining
 *                   only a long identifier like "--flag" or a comma-separated pair like "flag,f"
 *                   to define a long identifier like "--flag" together with a short identifier
 *                   like "-f"
 * @param helpText help-text for the argument for user-output
 *
 * @return false, if identifier is already registered or broken, else true
 */
bool
ArgParser::registerPlain(const std::string &identifier,
                         const std::string &helpText,
                         ErrorContainer &error)
{
    return registerArgument(identifier,
                            helpText,
                            ArgType::NO_TYPE,
                            false,
                            false,
                            false,
                            error);
}

/**
 * @brief register string-value
 *
 * @param identifier Identifier for the new argument. Its a single word like "flag" for defining
 *                   only a long identifier like "--flag" or a comma-separated pair like "flag,f"
 *                   to define a long identifier like "--flag" together with a short identifier
 *                   like "-f"
 * @param helpText help-text for the argument for user-output
 * @param required true, to make the argument required, fals to make it optional
 * @param withoutFlag true, the handle the identifier as flag for the value
 *
 * @return false, if identifier is already registered or broken, else true
 */
bool
ArgParser::registerString(const std::string &identifier,
                          const std::string &helpText,
                          ErrorContainer &error,
                          bool required,
                          bool withoutFlag)
{
    return registerArgument(identifier,
                            helpText,
                            ArgType::STRING_TYPE,
                            required,
                            withoutFlag,
                            true,
                            error);
}

/**
 * @brief register int/long value
 *
 * @param identifier Identifier for the new argument. Its a single word like "flag" for defining
 *                   only a long identifier like "--flag" or a comma-separated pair like "flag,f"
 *                   to define a long identifier like "--flag" together with a short identifier
 *                   like "-f"
 * @param helpText help-text for the argument for user-output
 * @param required true, to make the argument required, fals to make it optional
 * @param withoutFlag true, the handle the identifier as flag for the value. If this value is true
 *                    the required-value is set to true too.
 *
 * @return false, if identifier is already registered or broken, else true
 */
bool
ArgParser::registerInteger(const std::string &identifier,
                           const std::string &helpText,
                           ErrorContainer &error,
                           bool required,
                           bool withoutFlag)
{
    return registerArgument(identifier,
                            helpText,
                            ArgType::INT_TYPE,
                            required,
                            withoutFlag,
                            true,
                            error);
}


/**
 * @brief register float/double value
 *
 * @param identifier Identifier for the new argument. Its a single word like "flag" for defining
 *                   only a long identifier like "--flag" or a comma-separated pair like "flag,f"
 *                   to define a long identifier like "--flag" together with a short identifier
 *                   like "-f"
 * @param helpText help-text for the argument for user-output
 * @param required true, to make the argument required, fals to make it optional
 * @param withoutFlag true, the handle the identifier as flag for the value. If this value is true
 *                    the required-value is set to true too.
 *
 * @return false, if identifier is already registered or broken, else true
 */
bool
ArgParser::registerFloat(const std::string &identifier,
                         const std::string &helpText,
                         ErrorContainer &error,
                         bool required,
                         bool withoutFlag)
{
    return registerArgument(identifier,
                            helpText,
                            ArgType::FLOAT_TYPE,
                            required,
                            withoutFlag,
                            true,
                            error);
}

/**
 * @brief register bool value
 *
 * @param identifier Identifier for the new argument. Its a single word like "flag" for defining
 *                   only a long identifier like "--flag" or a comma-separated pair like "flag,f"
 *                   to define a long identifier like "--flag" together with a short identifier
 *                   like "-f"
 * @param helpText help-text for the argument for user-output
 * @param required true, to make the argument required, fals to make it optional
 * @param withoutFlag true, the handle the identifier as flag for the value. If this value is true
 *                    the required-value is set to true too.
 *
 * @return false, if identifier is already registered or broken, else true
 */
bool
ArgParser::registerBoolean(const std::string &identifier,
                           const std::string &helpText,
                           ErrorContainer &error,
                           bool required,
                           bool withoutFlag)
{
    return registerArgument(identifier,
                            helpText,
                            ArgType::BOOL_TYPE,
                            required,
                            withoutFlag,
                            true,
                            error);
}

/**
 * @brief register ne argument
 *
 * @param identifier Identifier for the new argument. Its a single word like "flag" for defining
 *                   only a long identifier like "--flag" or a comma-separated pair like "flag,f"
 *                   to define a long identifier like "--flag" together with a short identifier
 *                   like "-f"
 * @param helpText help-text for the argument for user-output
 * @param type type identifier
 * @param required true, to make the argument required, fals to make it optional
 * @param withoutFlag true, the handle the identifier as flag for the value. If this value is true
 *                    the required-value is set to true too.
 *
 * @return false, if identifier is already registered or broken, else true
 */
bool
ArgParser::registerArgument(const std::string &identifier,
                            const std::string &helpText,
                            const ArgType type,
                            bool required,
                            bool withoutFlag,
                            bool hasValue,
                            ErrorContainer &error)
{
    // precheck
    if(identifier.size() == 0
            || identifier.at(0) == ',')
    {
        error.addMeesage("No argument identifier was set");
        LOG_ERROR(error);
        return false;
    }

    ArgDefinition newArgument;

    // split identifier-string
    std::vector<std::string> identifierList;
    splitStringByDelimiter(identifierList, identifier, ',');

    // check splitting-result
    if(identifierList.size() > 2)
    {
        ErrorContainer error;
        error.addMeesage("argument identifier name is too long: " + identifier);
        LOG_ERROR(error);
        return false;
    }

    // prepare long identifier
    if(identifierList.at(0).size() == 0)
    {
        ErrorContainer error;
        error.addMeesage("argument identifier is invalid: " + identifier);
        LOG_ERROR(error);
        return false;
    }
    if(withoutFlag == false) {
        newArgument.longIdentifier = "--";
    }

    newArgument.longIdentifier += identifierList.at(0);

    // check if already used
    ArgParser::ArgDefinition* findLong = getArgument(newArgument.longIdentifier);
    if(findLong != nullptr)
    {
        error.addMeesage("argument already in use: " + newArgument.longIdentifier);
        LOG_ERROR(error);
        return false;
    }

    // prepare short identifier
    if(identifierList.size() == 2)
    {
        // check length
        if(identifierList.at(1).size() != 1)
        {
            error.addMeesage("Argument identifier is invalid: " + identifier);
            LOG_ERROR(error);
            return false;
        }

        newArgument.shortIdentifier = "-" + identifierList.at(1);

        // check if already used
        ArgParser::ArgDefinition* findShort = getArgument(newArgument.shortIdentifier);
        if(findShort != nullptr)
        {
            error.addMeesage("argument already in use: " + newArgument.shortIdentifier);
            LOG_ERROR(error);
            return false;
        }
    }

    // set other values
    newArgument.required = required;
    newArgument.hasValue = hasValue;
    newArgument.withoutFlag = withoutFlag;
    if(withoutFlag) {
        newArgument.required = true;
    }
    newArgument.type = type;
    newArgument.helpText = helpText;
    newArgument.results = new DataArray();

    m_argumentList.push_back(newArgument);

    return true;
}

/**
 * @brief convert argument-values
 *
 * @param value string-value
 * @param requiredType required type for the argument
 *
 * @return nullptr, if converting failed, else data-item with the converted value
 */
DataItem*
ArgParser::convertValue(const std::string &value,
                        const ArgParser::ArgType requiredType)
{
    // string value
    if(requiredType == ArgType::STRING_TYPE) {
        return new DataValue(value);
    }

    // long/int value
    if(requiredType == ArgType::INT_TYPE)
    {
        char* err = nullptr;
        const char* charValue = value.c_str();

        // convert to long-value
        const long longValue = std::strtol(charValue, &err, 10);
        if(std::string(err).size() != 0) {
            return nullptr;
        }

        return new DataValue(longValue);
    }

    // double/floag value
    if(requiredType == ArgType::FLOAT_TYPE)
    {
        char* err = nullptr;
        const char* charValue = value.c_str();

        // convert to double-value
        const double doubleValue = std::strtod(charValue, &err);
        if(std::string(err).size() != 0) {
            return nullptr;
        }

        return new DataValue(doubleValue);
    }

    // bool value
    if(requiredType == ArgType::BOOL_TYPE)
    {
        // convert true
        if(value == "true"
                || value == "True"
                || value == "1")
        {
            return new DataValue(true);
        }

        // convert false
        if(value == "false"
                || value == "False"
                || value == "0")
        {
            return new DataValue(false);
        }

        // if nothing match, it is no boolean
        return nullptr;
    }

    return nullptr;
}

/**
 * @brief precheck for the flags to handle --help and --version
 *
 * @param argc number of arguments
 * @param argv arguments
 *
 * @return true, if help or version flag was set, else true
 */
bool
ArgParser::precheckFlags(const int argc,
                         const char* argv[])
{
    const std::string programmPath(argv[0]);
    std::vector<std::string> pathParts;
    splitStringByDelimiter(pathParts, programmPath, '/');

    for(int i = 1; i < argc; i++)
    {
        const std::string currentArgument(argv[i]);

        // check for help-flag
        if(currentArgument == "-h"
                || currentArgument == "--help")
        {
            print(pathParts.at(pathParts.size()-1));
            return true;
        }

        if(m_version != "")
        {
            // check for version-flag
            if(currentArgument == "-v"
                    || currentArgument == "--version")
            {
                std::cout<<"version: "<<m_version<<std::endl;
                exit(0);
            }
        }
    }

    return false;
}

/**
 * @brief parse cli-arguments
 *
 * @param argc number of arguments
 * @param argv arguments
 *
 * @return false, if parsing failed
 */
bool
ArgParser::parse(const int argc,
                 char* argv[],
                 ErrorContainer &error)
{
    // TODO: find better solution without warning
    return parse(argc, (const char**)argv, error);
}

/**
 * @brief parse cli-arguments
 *
 * @param argc number of arguments
 * @param argv arguments as const char
 *
 * @return false, if parsing failed
 */
bool
ArgParser::parse(const int argc,
                 const char* argv[],
                 ErrorContainer &error)
{
    if(precheckFlags(argc, argv)) {
        return true;
    }

    m_positionCounter = 0;

    int i = 1;
    while(i < argc)
    {
        const std::string currentArgument(argv[i]);

        if(currentArgument.at(0) == '-')
        {
            // get and check identifier
            ArgParser::ArgDefinition* argIdent = getArgument(currentArgument);
            if(argIdent == nullptr)
            {
                error.addMeesage("unknown argument: " + currentArgument);
                LOG_ERROR(error);
                return false;
            }

            if(argIdent->hasValue)
            {
                // check if there is a value for the identifier
                if(i+1 == argc)
                {
                    error.addMeesage("flag has no value: " + currentArgument);
                    LOG_ERROR(error);
                    return false;
                }

                // get value
                const std::string currentValue(argv[i+1]);

                // convert value
                DataItem* convertedValue = convertValue(currentValue, argIdent->type);
                if(convertedValue == nullptr)
                {
                    const std::string errMsg = "argument has the false type: "
                                               "\n    required type: "
                                               + convertType(argIdent->type)
                                               + "\n    identifier: "
                                               + currentArgument
                                               + "\n    given value: "
                                               + currentValue;

                    error.addMeesage(errMsg);
                    LOG_ERROR(error);
                    return false;
                }

                // add converted value to results
                argIdent->results->append(convertedValue);

                i += 2;
            }
            else
            {
                i += 1;
            }

            argIdent->wasSet = true;
        }
        else
        {
            uint32_t counter = 0;
            for(uint64_t j = 0; j < m_argumentList.size(); j++)
            {
                if(m_argumentList[j].withoutFlag == true)
                {
                    ArgParser::ArgDefinition* argIdent = &m_argumentList[j];
                    if(m_positionCounter == counter)
                    {
                        // convert value
                        DataItem* convertedValue = convertValue(currentArgument, argIdent->type);
                        if(convertedValue == nullptr)
                        {
                            const std::string errMsg = "argument has the false type: "
                                                       "\n    required type: "
                                                       + convertType(argIdent->type)
                                                       + "\n    identifier: "
                                                       + m_argumentList[j].longIdentifier
                                                       + "\n    given value: "
                                                       + currentArgument;

                            error.addMeesage(errMsg);
                            LOG_ERROR(error);
                            return false;
                        }

                        // add converted value to results
                        argIdent->results->append(convertedValue);

                        // update counter
                        m_positionCounter++;
                        j = m_argumentList.size();
                    }

                    counter++;
                }
            }

            i += 1;
        }
    }

    // check if all requirements are satisfied
    for(uint32_t i = 0; i < m_argumentList.size(); i++)
    {
        if(m_argumentList[i].results->size() == 0
                && m_argumentList[i].required)
        {
            error.addMeesage("argument is required but was not set: "
                             + m_argumentList[i].longIdentifier);
            LOG_ERROR(error);
            return false;
        }
    }

    return true;
}

/**
 * @brief check how often a flag was set
 *
 * @param identifier identifier-name of the flag
 *
 * @return 0, if the flag was not set, else the counter of how often the flag was set
 *
 */
uint64_t
ArgParser::getNumberOfValues(const std::string &identifier)
{
    ArgParser::ArgDefinition* arg = getArgument(identifier);
    if(arg == nullptr) {
        return 0;
    }

    return arg->results->size();
}

/**
 * @brief check if a flag was set
 *
 * @param identifier identifier-name of the flag
 *
 * @return true, if flag was set, else false
 */
bool
ArgParser::wasSet(const std::string &identifier)
{
    ArgParser::ArgDefinition* arg = getArgument(identifier);
    if(arg == nullptr) {
        return false;
    }

    return arg->wasSet;
}

/**
 * @brief get parsed string values of an identifier
 *
 * @param identifier regested identifier without "--" or "-"
 *
 * @return list of parsed string values
 */
const std::vector<std::string>
ArgParser::getStringValues(const std::string &identifier)
{
    std::vector<std::string> result;

    // get registered argument
    ArgParser::ArgDefinition* arg = getArgument(identifier);
    if(arg == nullptr) {
        return result;
    }

    // check argument-type
    if(arg->type != ArgType::STRING_TYPE) {
        return result;
    }

    // build list with all results
    for(uint32_t i = 0; i < arg->results->size(); i++) {
        result.push_back(arg->results->get(i)->getString());
    }

    return result;
}

/**
 * @brief get parsed long values of an identifier
 *
 * @param identifier regested identifier without "--" or "-"
 *
 * @return list of parsed long values
 */
const std::vector<long>
ArgParser::getIntValues(const std::string &identifier)
{
    std::vector<long> result;

    // get registered argument
    ArgParser::ArgDefinition* arg = getArgument(identifier);
    if(arg == nullptr) {
        return result;
    }

    // check argument-type
    if(arg->type != ArgType::INT_TYPE) {
        return result;
    }

    // build list with all results
    for(uint32_t i = 0; i < arg->results->size(); i++) {
        result.push_back(arg->results->get(i)->getLong());
    }

    return result;
}

/**
 * @brief get parsed double values of an identifier
 *
 * @param identifier regested identifier without "--" or "-"
 *
 * @return list of parsed double values
 */
const std::vector<double>
ArgParser::getFloatValues(const std::string &identifier)
{
    std::vector<double> result;

    // get registered argument
    ArgParser::ArgDefinition* arg = getArgument(identifier);
    if(arg == nullptr) {
        return result;
    }

    // check argument-type
    if(arg->type != ArgType::FLOAT_TYPE) {
        return result;
    }

    // build list with all results
    for(uint32_t i = 0; i < arg->results->size(); i++) {
        result.push_back(arg->results->get(i)->getDouble());
    }

    return result;
}

/**
 * @brief get parsed bool values of an identifier
 *
 * @param identifier regested identifier without "--" or "-"
 *
 * @return list of parsed bool values
 */
const std::vector<bool>
ArgParser::getBoolValues(const std::string &identifier)
{
    std::vector<bool> result;

    // get registered argument
    ArgParser::ArgDefinition* arg = getArgument(identifier);
    if(arg == nullptr) {
        return result;
    }

    // check argument-type
    if(arg->type != ArgType::BOOL_TYPE) {
        return result;
    }

    // build list with all results
    for(uint32_t i = 0; i < arg->results->size(); i++) {
        result.push_back(arg->results->get(i)->getBool());
    }

    return result;
}

/**
 * @brief get parsed string value of an required identifier
 *
 * @param identifier regested identifier
 *
 * @return parsed string value
 */
const std::string
ArgParser::getStringValue(const std::string &identifier)
{
    std::vector<std::string> result;

    // get registered argument
    ArgParser::ArgDefinition* arg = getArgument(identifier);
    if(arg == nullptr) {
        return "";
    }

    // check argument-type
    if(arg->type != ArgType::STRING_TYPE) {
        return "";
    }

    // check result not empty
    if(arg->results->size() == 0) {
        return "";
    }

    return arg->results->get(0)->getString();
}

/**
 * @brief get parsed long value of an required identifier
 *
 * @param identifier regested identifier
 *
 * @return parsed long value
 */
long
ArgParser::getIntValue(const std::string &identifier)
{
    std::vector<long> result;

    // get registered argument
    ArgParser::ArgDefinition* arg = getArgument(identifier);
    if(arg == nullptr) {
        return 0l;
    }

    // check argument-type
    if(arg->type != ArgType::INT_TYPE) {
        return 0l;
    }

    // check result not empty
    if(arg->results->size() == 0) {
        return 0l;
    }

    return arg->results->get(0)->getLong();
}

/**
 * @brief get parsed double value of an required identifier
 *
 * @param identifier regested identifier
 *
 * @return parsed double value
 */
double
ArgParser::getFloatValue(const std::string &identifier)
{
    std::vector<double> result;

    // get registered argument
    ArgParser::ArgDefinition* arg = getArgument(identifier);
    if(arg == nullptr) {
        return 0.0;
    }

    // check argument-type
    if(arg->type != ArgType::FLOAT_TYPE) {
        return 0.0;
    }

    // check result not empty
    if(arg->results->size() == 0) {
        return 0.0;
    }

    return arg->results->get(0)->getDouble();
}

/**
 * @brief get parsed bool value of an required identifier
 *
 * @param identifier regested identifier
 *
 * @return parsed bool value
 */
bool
ArgParser::getBoolValue(const std::string &identifier)
{
    std::vector<bool> result;

    // get registered argument
    ArgParser::ArgDefinition* arg = getArgument(identifier);
    if(arg == nullptr) {
        return false;
    }

    // check argument-type
    if(arg->type != ArgType::BOOL_TYPE) {
        return false;
    }

    // check result not empty
    if(arg->results->size() == 0) {
        return false;
    }

    return arg->results->get(0)->getBool();
}

/**
 * @brief convert type-value into string
 *
 * @param type type-value
 *
 * @return converted string
 */
const std::string
ArgParser::convertType(ArgParser::ArgType type)
{
    if(type == ArgType::STRING_TYPE) {
        return "string";
    }
    if(type == ArgType::INT_TYPE) {
        return "number";
    }
    if(type == ArgType::FLOAT_TYPE) {
        return "floating point";
    }
    if(type == ArgType::BOOL_TYPE) {
        return "boolean";
    }

    return "";
}

/**
 * @brief print arguments on cli
 */
void
ArgParser::print(const std::string &commandName)
{
    std::string commandString = commandName + " [options]";

    // prepare table for arguments with flags
    TableItem withFlags;
    withFlags.addColumn("long");
    withFlags.addColumn("short");
    withFlags.addColumn("type");
    withFlags.addColumn("is required");
    withFlags.addColumn("help-text");

    for(uint32_t i = 0; i < m_argumentList.size(); i++)
    {
        if(m_argumentList.at(i).withoutFlag == false)
        {
            // get type
            const std::string type = convertType(m_argumentList.at(i).type);

            // required flag
            std::string required = "";
            if(m_argumentList.at(i).required)
            {
                required = "x";
                commandString += " " + m_argumentList.at(i).longIdentifier + " ...";
            }

            // set row of table
            withFlags.addRow(std::vector<std::string>{
                m_argumentList.at(i).longIdentifier,
                m_argumentList.at(i).shortIdentifier,
                type,
                required,
                m_argumentList.at(i).helpText
            });
        }
    }

    // prepare table for arguments without flags
    TableItem withoutFlags;
    withoutFlags.addColumn("name");
    withoutFlags.addColumn("type");
    withoutFlags.addColumn("text");

    for(uint32_t i = 0; i < m_argumentList.size(); i++)
    {
        if(m_argumentList.at(i).withoutFlag == true)
        {
            // get type
            const std::string type = convertType(m_argumentList.at(i).type);

            // set row of table
            withoutFlags.addRow(std::vector<std::string>{
                "<" + m_argumentList.at(i).longIdentifier + ">",
                type,
                m_argumentList.at(i).helpText
            });

            commandString += " <" + m_argumentList.at(i).longIdentifier + ">";
        }
    }

    std::cout<<"command: "<<commandString<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Options:"<<std::endl;
    std::cout<<withFlags.toString(200)<<std::endl;
    std::cout<<"Required:"<<std::endl;
    std::cout<<withoutFlags.toString(200)<<std::endl;

    exit(0);
}

/**
 * @brief get a registered argument
 *
 * @param identifier identifier of the argument without "--" or "-"
 *
 * @return nullptr, if identifier is unknown, else pointer to the registered argument
 */
ArgParser::ArgDefinition*
ArgParser::getArgument(const std::string &identifier)
{
    for(uint32_t i = 0; i < m_argumentList.size(); i++)
    {
        if(m_argumentList.at(i).longIdentifier == identifier
                || m_argumentList.at(i).shortIdentifier == identifier
                || m_argumentList.at(i).longIdentifier == "--" + identifier
                || m_argumentList.at(i).shortIdentifier == "-" + identifier)
        {
            return &m_argumentList.at(i);
        }
    }

    return nullptr;
}

}
