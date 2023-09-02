/**
 * @file       cluster_parser_interface.h
 *
 * @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright  Apache License Version 2.0
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

#ifndef KITSUNEMIMI_HANAMI_SEGMENT_PARSER_PARSER_INTERFACE_H
#define KITSUNEMIMI_HANAMI_SEGMENT_PARSER_PARSER_INTERFACE_H

#include <iostream>
#include <mutex>

#include <hanami_common/logger.h>

namespace Kitsunemimi::Hanami
{
struct ClusterMeta;
class location;

class ClusterParserInterface
{

public:
    static ClusterParserInterface* getInstance();
    ~ClusterParserInterface();

    // connection the the scanner and parser
    void scan_begin(const std::string &inputString);
    void scan_end();
    bool parse(ClusterMeta* result,
               const std::string &inputString,
               ErrorContainer &error);
    const std::string removeQuotes(const std::string &input);

    // Error handling.
    void error(const Kitsunemimi::Hanami::location &location,
               const std::string& message);

    ClusterMeta* output = nullptr;

private:
    ClusterParserInterface(const bool traceParsing = false);

    static ClusterParserInterface* m_instance;

    std::string m_errorMessage = "";
    std::string m_inputString = "";
    std::mutex m_lock;

    bool m_traceParsing = false;
};

}

#endif // KITSUNEMIMI_HANAMI_SEGMENT_PARSER_PARSER_INTERFACE_H
