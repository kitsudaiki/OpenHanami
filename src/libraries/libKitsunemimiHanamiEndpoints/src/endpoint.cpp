/**
 * @file        endpoint.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2021 Tobias Anker
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

#include <libKitsunemimiHanamiEndpoints/endpoint.h>
#include <endpoint_parsing/endpoint_parser_interface.h>

namespace Kitsunemimi
{
namespace Hanami
{

Endpoint* Endpoint::m_endpoints = nullptr;

/**
 * @brief Endpoint::Endpoint
 */
Endpoint::Endpoint() {}

/**
 * @brief Endpoint::~Endpoint
 */
Endpoint::~Endpoint() {}

/**
 * @brief get instance, which must be already initialized
 *
 * @return instance-object
 */
Endpoint*
Endpoint::getInstance()
{
    if(m_endpoints == nullptr) {
        m_endpoints = new Endpoint();
    }
    return m_endpoints;
}

/**
 * @brief parse endpoint-file content
 *
 * @param input input-string with endpoint-definition to parse
 * @param error reference for error-output
 *
 * @return true, if parsing was successful, else false
 */
bool
Endpoint::parse(const std::string &input,
                ErrorContainer &error)
{
    EndpointParserInterface* parser = EndpointParserInterface::getInstance();
    return parser->parse(&endpointRules, input, error);
}

/**
 * @brief map the endpoint to the real target
 *
 * @param result reference to the result to identify the target
 * @param id request-id
 * @param type requested http-request-type
 *
 * @return false, if mapping failes, else true
 */
bool
Endpoint::mapEndpoint(EndpointEntry &result,
                      const std::string &id,
                      const HttpRequestType type)
{
    std::map<std::string, std::map<HttpRequestType, EndpointEntry>>::const_iterator id_it;
    id_it = endpointRules.find(id);

    if(id_it != endpointRules.end())
    {
        std::map<HttpRequestType, EndpointEntry>::const_iterator type_it;
        type_it = id_it->second.find(type);

        if(type_it != id_it->second.end())
        {
            result.type = type_it->second.type;
            result.group = type_it->second.group;
            result.name = type_it->second.name;
            return true;
        }
    }

    return false;
}

/**
 * @brief add new custom-endpoint without the parser
 *
 * @param id identifier for the new entry
 * @param httpType http-type (get, post, put, delete)
 * @param sakuraType sakura-type (tree or blossom)
 * @param group blossom-group
 * @param name tree- or blossom-id
 *
 * @return false, if id together with http-type is already registered, else true
 */
bool
Endpoint::addEndpoint(const std::string &id,
                      const HttpRequestType &httpType,
                      const SakuraObjectType &sakuraType,
                      const std::string &group,
                      const std::string &name)
{
    EndpointEntry newEntry;
    newEntry.type = sakuraType;
    newEntry.group = group;
    newEntry.name = name;

    // search for id
    std::map<std::string, std::map<HttpRequestType, EndpointEntry>>::iterator id_it;
    id_it = endpointRules.find(id);
    if(id_it != endpointRules.end())
    {
        // search for http-type
        std::map<HttpRequestType, EndpointEntry>::iterator type_it;
        type_it = id_it->second.find(httpType);
        if(type_it != id_it->second.end()) {
            return false;
        }

        // add new
        id_it->second.emplace(httpType, newEntry);
    }
    else
    {
        // add new
        std::map<HttpRequestType, EndpointEntry> typeEntry;
        typeEntry.emplace(httpType, newEntry);
        endpointRules.emplace(id, typeEntry);
    }

    return true;
}

}  // namespace Hanami
}  // namespace Kitsunemimi
