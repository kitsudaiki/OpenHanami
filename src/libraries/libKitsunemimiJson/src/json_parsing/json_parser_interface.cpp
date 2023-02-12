/**
 *  @file    json_parser_interface.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <json_parsing/json_parser_interface.h>
#include <json_parser.h>

#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiCommon/items/data_items.h>

using Kitsunemimi::DataItem;
using Kitsunemimi::DataArray;
using Kitsunemimi::DataValue;
using Kitsunemimi::DataMap;

# define YY_DECL \
    Kitsunemimi::JsonParser::symbol_type jsonlex (Kitsunemimi::JsonParserInterface& driver)
YY_DECL;

namespace Kitsunemimi
{

Kitsunemimi::JsonParserInterface* JsonParserInterface::m_instance = nullptr;

using Kitsunemimi::splitStringByDelimiter;

/**
 * @brief The class is the interface for the bison-generated parser.
 *        It starts the parsing-process and store the returned values.
 *
 * @param traceParsing If set to true, the scanner prints all triggered rules.
 *                     It is only for better debugging.
 */
JsonParserInterface::JsonParserInterface(const bool traceParsing)
{
    m_traceParsing = traceParsing;
}

/**
 * @brief static methode to get instance of the interface
 *
 * @return pointer to the static instance
 */
JsonParserInterface*
JsonParserInterface::getInstance()
{
    if(m_instance == nullptr) {
        m_instance = new JsonParserInterface();
    }

    return m_instance;
}

/**
 * @brief destructor
 */
JsonParserInterface::~JsonParserInterface()
{
    if(m_output != nullptr) {
        delete m_output;
    }
}

/**
 * @brief parse string
 *
 * @param inputString string which should be parsed
 * @param reference for error-message
 *
 * @return resulting object
 */
DataItem*
JsonParserInterface::parse(const std::string &inputString,
                           ErrorContainer &error)
{
    DataItem* result = nullptr;

    std::lock_guard<std::mutex> guard(m_lock);

    // init global values
    m_inputString = inputString;
    m_errorMessage = "";
    int parserResult = 0;
    Kitsunemimi::JsonParser parser(*this);

    // 1. dry-run to check syntax
    dryRun = true;
    this->scan_begin(inputString);
    parserResult = parser.parse();
    this->scan_end();

    // handle negative result
    if(parserResult != 0
            || m_errorMessage.size() > 0)
    {
        error.addMeesage(m_errorMessage);
        LOG_ERROR(error);
        return nullptr;
    }

    // 2. real run on the valid string
    dryRun = false;
    this->scan_begin(inputString);
    parser.parse();
    this->scan_end();

    result = m_output;
    m_output = nullptr;

    return result;
}

/**
 * @brief remove quotes at the beginning and end of a string
 *
 * @param input input-string
 *
 * @return cleared string
 */
const std::string
JsonParserInterface::removeQuotes(const std::string &input)
{
    // precheck
    if(input.length() == 0) {
        return input;
    }

    // clear
    if(input[0] == '\"'
            && input[input.length()-1] == '\"')
    {
        std::string result = "";
        for(uint32_t i = 1; i < input.length()-1; i++) {
            result += input[i];
        }

        return result;
    }

    return input;
}

/**
 * @brief Is called for the parser after successfully parsing the input-string
 *
 * @param output parser-output as data-item
 */
void
JsonParserInterface::setOutput(DataItem* output)
{
     m_output = output;
}

/**
 * @brief Is called from the parser in case of an error
 *
 * @param location location-object of the bison-parser,
 *                 which contains the informations of the location
 *                 of the syntax-error in the parsed string
 * @param message error-specific message from the parser
 */
void
JsonParserInterface::error(const Kitsunemimi::location& location,
                           const std::string& message)
{
    if(m_errorMessage.size() > 0) {
        return;
    }

    // get the broken part of the parsed string
    const uint32_t errorStart = location.begin.column;
    const uint32_t errorLength = location.end.column - location.begin.column;
    const uint32_t linenumber = location.begin.line;

    std::vector<std::string> splittedContent;
    splitStringByDelimiter(splittedContent, m_inputString, '\n');

    // build error-message
    m_errorMessage =  "ERROR while parsing json-formated string \n";
    m_errorMessage += "parser-message: " + message + " \n";
    m_errorMessage += "line-number: " + std::to_string(linenumber) + " \n";

    if(splittedContent[linenumber - 1].size() > errorStart - 1 + errorLength)
    {
        m_errorMessage.append("position in line: " +  std::to_string(location.begin.column) + "\n");
        m_errorMessage.append("broken part in string: \""
                              + splittedContent[linenumber - 1].substr(errorStart - 1, errorLength)
                              + "\"");
    }
    else
    {
        m_errorMessage.append("position in line: UNKNOWN POSITION (maybe a string was not closed)");
    }
}

}
