/**
 * @file        http_server.cpp
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

#include "http_server.h"

#include <hanami_root.h>

#include <api/endpoint_processing/http_processing/http_processing.h>
#include <api/endpoint_processing/http_websocket_thread.h>

#include <hanami_common/logger.h>
#include <hanami_common/files/text_file.h>

/**
 * @brief constructor
 *
 * @param address listen address for the server
 * @param port listen port for the server
 * @param certFilePath path to cert-file
 * @param keyFilePath path to key-file
 */
HttpServer::HttpServer(const std::string &address,
                       const uint16_t port)
    : Hanami::Thread("HttpServer"),
      //m_ctx{boost::asio::ssl::context::tlsv13_server},
      m_address(address),
      m_port(port)
{}

/**
 * @brief run server-thread
 */
void
HttpServer::run()
{
    Hanami::ErrorContainer error;

    LOG_INFO("start HTTP-server on address "
             + m_address
             + " and port "
             + std::to_string(m_port));
    try
    {
        // create server
        const net::ip::address address = net::ip::make_address(m_address);
        net::io_context ioc{1};
        tcp::acceptor acceptor{ioc, {address, m_port}};

        while(m_abort == false)
        {
            // create socket-object for incoming connection
            tcp::socket* socket = new tcp::socket{ioc};
            acceptor.accept(*socket);
            addSocket(socket);
        }
    }
    catch (const std::exception& e)
    {
        error.addMeesage("Error-message while running http-server: '"
                         + std::string(e.what())
                         + "'");
        LOG_ERROR(error);
    }
}

/**
 * @brief get socket from the queue
 *
 * @return socket
 */
tcp::socket*
HttpServer::getSocket()
{
    std::lock_guard<std::mutex> guard(m_queueMutex);

    tcp::socket* result = nullptr;
    if(m_queue.size() > 0)
    {
        result = m_queue.front();
        m_queue.pop_front();
    }

    return result;
}

/**
 * @brief add socket to queue
 *
 * @param socket new socket for the queue
 */
void
HttpServer::addSocket(tcp::socket* socket)
{
    std::lock_guard<std::mutex> guard(m_queueMutex);
    m_queue.push_back(socket);
}
