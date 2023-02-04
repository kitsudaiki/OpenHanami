/**
 * @file        endpoint.h
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

#ifndef ENDPOINT_H
#define ENDPOINT_H

#include <string>
#include <map>
#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{
namespace Hanami
{
class Endpoint_Test;

struct EndpointEntry
{
    SakuraObjectType type = BLOSSOM_TYPE;
    std::string group = "-";
    std::string name = "";
};

class Endpoint
{
public:
    static Endpoint* getInstance();

    ~Endpoint();

    bool parse(const std::string &input, ErrorContainer &error);

    bool mapEndpoint(EndpointEntry &result,
                     const std::string &id,
                     const HttpRequestType type);
    bool addEndpoint(const std::string &id,
                     const HttpRequestType &httpType,
                     const SakuraObjectType &sakuraType,
                     const std::string &group,
                     const std::string &name);

    std::map<std::string, std::map<HttpRequestType, EndpointEntry>> endpointRules;

private:
    Endpoint();

    static Endpoint* m_endpoints;

    friend Endpoint_Test;
};

}  // namespace Hanami
}  // namespace Kitsunemimi

#endif // ENDPOINT_H
