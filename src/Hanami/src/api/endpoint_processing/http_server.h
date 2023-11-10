/**
 * @file        http_server.h
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

#ifndef TORIIGATEWAY_HTTP_SERVER_H
#define TORIIGATEWAY_HTTP_SERVER_H

#include <hanami_common/logger.h>
#include <hanami_common/threading/thread.h>

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/version.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

using tcp = boost::asio::ip::tcp;

class HttpProcessing;

class HttpServer : public Hanami::Thread
{
   public:
    HttpServer(const std::string& address, const uint16_t port);

    HttpProcessing* httpProcessing = nullptr;

    tcp::socket* getSocket();
    void addSocket(tcp::socket* socket);

   protected:
    void run();

   private:
    const std::string m_address = "";
    const uint16_t m_port = 0;

    std::deque<tcp::socket*> m_queue;
    std::mutex m_queueMutex;
};

#endif  // TORIIGATEWAY_HTTP_SERVER_H
