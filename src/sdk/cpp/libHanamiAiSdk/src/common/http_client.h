/**
 * @file        request.h
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

#ifndef KITSUNEMIMI_HANAMISDK_HANAMI_REQUEST_H
#define KITSUNEMIMI_HANAMISDK_HANAMI_REQUEST_H

#include <cstdlib>
#include <iostream>
#include <string>

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/error.hpp>
#include <boost/asio/ssl/stream.hpp>

namespace beast = boost::beast; // from <boost/beast.hpp>
namespace http = beast::http;   // from <boost/beast/http.hpp>

namespace net = boost::asio;    // from <boost/asio.hpp>
namespace ssl = net::ssl;       // from <boost/asio/ssl.hpp>
using tcp = net::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

#include <hanami_common/logger.h>

namespace HanamiAI
{

class HanamiRequest
{
public:
    static HanamiRequest* getInstance();
    ~HanamiRequest();

    bool init(const std::string &host = "",
              const std::string &port = "",
              const std::string &user = "",
              const std::string &password = "",
              const std::string &cacert = "");

    bool sendGetRequest(std::string &response,
                        const std::string &path,
                        const std::string &vars,
                        Hanami::ErrorContainer &error);

    bool sendPostRequest(std::string &response,
                         const std::string &path,
                         const std::string &vars,
                         const std::string &body,
                         Hanami::ErrorContainer &error);

    bool sendPutRequest(std::string &response,
                        const std::string &path,
                        const std::string &vars,
                        const std::string &body,
                        Hanami::ErrorContainer &error);

    bool sendDeleteRequest(std::string &response,
                           const std::string &path,
                           const std::string &vars,
                           Hanami::ErrorContainer &error);

    const std::string& getToken() const;
    const std::string& getPort() const;
    const std::string& getHost() const;

    void updateToken(const std::string &newToken);

private:
    HanamiRequest();
    static HanamiRequest* m_instance;

    std::string m_host = "";
    std::string m_port = "";
    std::string m_cacert = "";
    std::string m_token = "";
    std::string m_userId = "";
    std::string m_password = "";

    bool requestToken(Hanami::ErrorContainer &error);
    bool makeRequest(std::string &response,
                     const http::verb type,
                     const std::string &path,
                     const std::string &vars,
                     const std::string &jsonBody,
                     Hanami::ErrorContainer &error);
    uint16_t makeSingleRequest(std::string &response,
                               const http::verb type,
                               const std::string &target,
                               const std::string &jsonBody,
                               Hanami::ErrorContainer &error);
    bool getEnvVar(std::string &content,
                   const std::string &key) const;
};

} // namespace HanamiAI

#endif // KITSUNEMIMI_HANAMISDK_HANAMI_REQUEST_H
