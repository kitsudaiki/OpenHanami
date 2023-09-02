/**
 * @file        init.cpp
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

#include <hanami_sdk/init.h>
#include <common/http_client.h>

namespace Hanami
{

/**
 * @brief initialize new client-session by requesting a access-token from the server
 *
 * @param host target-address to connect
 * @param port target-port to connect
 * @param user name of the user
 * @param password password of the user
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
initClient(const std::string &host,
           const std::string &port,
           const std::string &user,
           const std::string &password,
           Hanami::ErrorContainer &error)
{
    HanamiRequest* request = HanamiRequest::getInstance();
    if(request->init(host, port, user, password) == false)
    {
        error.addMeesage("Failed to initialize hanami-client");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

} // namespace Hanami
