/**
 * @file        websocket_client.h
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

#ifndef WEBSOCKETCLIENT_H
#define WEBSOCKETCLIENT_H

#include <libKitsunemimiCommon/logger.h>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

namespace HanamiAI
{

class WebsocketClient
{
public:
    WebsocketClient();
    ~WebsocketClient();

    bool initClient(std::string &socketUuid,
                    const std::string &token,
                    const std::string &target,
                    const std::string &host,
                    const std::string &port,
                    Kitsunemimi::ErrorContainer &error);
    bool sendMessage(const void* data,
                     const uint64_t dataSize,
                     Kitsunemimi::ErrorContainer &error);

    uint8_t* readMessage(uint64_t &numberOfByes,
                         Kitsunemimi::ErrorContainer &error);

private:
    websocket::stream<beast::ssl_stream<tcp::socket>>* m_websocket = nullptr;
    bool loadCertificates(boost::asio::ssl::context &ctx);
};

}  // namespace HanamiAI

#endif // WEBSOCKETCLIENT_H
