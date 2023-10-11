/**
 *  @file    ini_parser_interface.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef INIPARSERINTERFACE_H
#define INIPARSERINTERFACE_H

#include <hanami_common/logger.h>

#include <map>
#include <mutex>
#include <string>
#include <vector>

using std::map;
using std::pair;
using std::string;

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace Hanami
{
class location;

class IniParserInterface
{
   public:
    static IniParserInterface *getInstance();

    // connection the the scanner and parser
    void scan_begin(const std::string &inputString);
    void scan_end();
    json parse(const std::string &inputString, ErrorContainer &error);
    const std::string removeQuotes(const std::string &input);

    // output-handling
    void setOutput(json output);

    // Error handling.
    void error(const Hanami::location &location, const std::string &message);

    // static variables, which are used in lexer and parser
    static bool m_outsideComment;

   private:
    IniParserInterface(const bool traceParsing = false);

    static IniParserInterface *m_instance;

    json m_output;
    std::string m_errorMessage = "";
    std::string m_inputString = "";
    std::mutex m_lock;

    bool m_traceParsing = false;
};

}  // namespace Hanami

#endif  // INIPARSERINTERFACE_H
