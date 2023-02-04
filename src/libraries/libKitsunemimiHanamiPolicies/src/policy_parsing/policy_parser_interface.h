/**
 * @file        policy_parser_interface.h
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
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

#ifndef POLICY_PARSER_INTERFACE_H
#define POLICY_PARSER_INTERFACE_H

#include <iostream>
#include <mutex>
#include <map>
#include <string>
#include <vector>
#include <libKitsunemimiHanamiPolicies/policy.h>

namespace Kitsunemimi
{
namespace Hanami
{
class location;

class PolicyParserInterface
{

public:
    static PolicyParserInterface* getInstance();
    ~PolicyParserInterface();

    // connection the the scanner and parser
    void scan_begin(const std::string &inputString);
    void scan_end();
    bool parse(std::map<std::string, std::map<std::string, PolicyEntry>>* result,
               const std::string &inputString,
               ErrorContainer &error);
    const std::string removeQuotes(const std::string &input);

    std::map<std::string, std::map<std::string, PolicyEntry>>* m_result = nullptr;

    // Error handling.
    void error(const location &location,
               const std::string& message);

private:
    PolicyParserInterface(const bool traceParsing = false);

    static PolicyParserInterface* m_instance;

    std::string m_errorMessage = "";
    std::string m_inputString = "";
    std::mutex m_lock;

    bool m_traceParsing = false;
};

}  // namespace Policy
}  // namespace Kitsunemimi

#endif // POLICY_PARSER_INTERFACE_H
