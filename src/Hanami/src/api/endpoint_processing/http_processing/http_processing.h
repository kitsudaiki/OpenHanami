/**
 * @file        http_processing.h
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

#ifndef TORIIGATEWAY_HTTP_PROCESSING_H
#define TORIIGATEWAY_HTTP_PROCESSING_H

#include <common.h>
#include <hanami_common/logger.h>
#include <hanami_policies/policy.h>

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/version.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/config.hpp>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace net = boost::asio;             // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>
namespace websocket = beast::websocket;  // from <boost/beast/websocket.hpp>
namespace ssl = boost::asio::ssl;        // from <boost/asio/ssl.hpp>

class Blossom;

class HttpProcessing
{
   public:
    bool processRequest(http::request<http::string_body>& httpRequest,
                        http::response<http::dynamic_body>& httpResponse,
                        Hanami::ErrorContainer& error);

    // endpoints
    bool mapEndpoint(EndpointEntry& result,
                     const std::string& id,
                     const Hanami::HttpRequestType type);
    bool addEndpoint(const std::string& id,
                     const Hanami::HttpRequestType& httpType,
                     const SakuraObjectType& sakuraType,
                     const std::string& group,
                     const std::string& name);

    // blossoms
    bool triggerBlossom(json& result,
                        const std::string& blossomName,
                        const std::string& blossomGroupName,
                        const json& context,
                        const json& initialValues,
                        BlossomStatus& status,
                        Hanami::ErrorContainer& error);
    bool doesBlossomExist(const std::string& groupName, const std::string& itemName);
    bool addBlossom(const std::string& groupName, const std::string& itemName, Blossom* newBlossom);
    Blossom* getBlossom(const std::string& groupName, const std::string& itemName);

    std::map<std::string, std::map<Hanami::HttpRequestType, EndpointEntry>> endpointRules;

   private:
    bool processControlRequest(http::response<http::dynamic_body>& httpResponse,
                               const std::string& uri,
                               const std::string& token,
                               const std::string& inputValues,
                               const Hanami::HttpRequestType httpType,
                               Hanami::ErrorContainer& error);
    bool checkStatusCode(Blossom* blossom,
                         const std::string& blossomName,
                         const std::string& blossomGroupName,
                         BlossomStatus& status,
                         Hanami::ErrorContainer& error);

    std::map<std::string, std::map<std::string, Blossom*>> m_registeredBlossoms;
};

#endif  // TORIIGATEWAY_HTTP_PROCESSING_H
