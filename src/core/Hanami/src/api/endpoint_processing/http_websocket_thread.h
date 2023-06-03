/**
 * @file        http_websocket_thread.h
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

#ifndef TORIIGATEWAY_HTTP_THREAD_H
#define TORIIGATEWAY_HTTP_THREAD_H

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/config.hpp>
#include <boost/beast/websocket.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <filesystem>

#include <hanami_messages.proto3.pb.h>
#include <libKitsunemimiCommon/threading/thread.h>
#include <libKitsunemimiCommon/logger.h>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>

class Cluster;

class HttpWebsocketThread
        : public Kitsunemimi::Thread
{
public:
    HttpWebsocketThread(const std::string &threadName);

    bool sendData(const void* data, const uint64_t dataSize, const bool waitForInput);
    Cluster* m_targetCluster = nullptr;
    void closeClient(Kitsunemimi::ErrorContainer &error);

protected:
    void run();

private:
    bool handleSocket(tcp::socket* socket,
                      Kitsunemimi::ErrorContainer &error);
    bool readMessage(beast::ssl_stream<tcp::socket&> &stream,
                     http::request<http::string_body> &httpRequest,
                     Kitsunemimi::ErrorContainer &error);
    bool sendResponse(beast::ssl_stream<tcp::socket&> &stream,
                      http::response<http::dynamic_body> &httpResponse,
                      Kitsunemimi::ErrorContainer &error);

    // websocket-functions and variables
    bool initWebsocket(http::request<http::string_body> &httpRequest);
    void runWebsocket();
    bool processInitialMessage(const std::string &message,
                               Kitsunemimi::ErrorContainer &error);

    websocket::stream<beast::ssl_stream<tcp::socket&>>* m_webSocket = nullptr;
    std::string m_uuid = "";
    std::string m_target = "";
    bool m_waitForInput = true;
    bool m_websocketClosed = true;
    bool m_clientInit = false;
    bool m_datasetConnection = false;
    bool m_clusterConnection = false;
};

#endif // TORIIGATEWAY_HTTP_THREAD_H
