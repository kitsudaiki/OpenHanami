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
                       const uint16_t port,
                       const std::string &certFilePath,
                       const std::string &keyFilePath)
    : Hanami::Thread("HttpServer"),
      m_ctx{boost::asio::ssl::context::tlsv13_server},
      m_address(address),
      m_port(port),
      m_certFilePath(certFilePath),
      m_keyFilePath(keyFilePath)
{}

/**
 * @brief load certificates from files into ssl-context
 *
 * @param ctx ssl-context
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HttpServer::loadCertificates(boost::asio::ssl::context &ctx,
                             Hanami::ErrorContainer &error)
{
    std::string errorMessage = "";
    std::string cert = "";
    std::string key = "";
    bool ret = false;

    // read certificate
    ret = Hanami::readFile(cert, m_certFilePath, error);
    if(ret == false)
    {
        error.addMeesage("failed to read certificate-file: '" + m_certFilePath + "'");
        return false;
    }

    // read key
    ret = Hanami::readFile(key, m_keyFilePath, error);
    if(ret == false)
    {
        error.addMeesage("failed to read key-file: '" + m_certFilePath + "'");
        return false;
    }

    // TODO: replace hard-coded string by a file
    /*const std::string dh = "-----BEGIN DH PARAMETERS-----\n"
                           "MIIBCAKCAQEArzQc5mpm0Fs8yahDeySj31JZlwEphUdZ9StM2D8+Fo7TMduGtSi+\n"
                           "/HRWVwHcTFAgrxVdm+dl474mOUqqaz4MpzIb6+6OVfWHbQJmXPepZKyu4LgUPvY/\n"
                           "4q3/iDMjIS0fLOu/bLuObwU5ccZmDgfhmz1GanRlTQOiYRty3FiOATWZBRh6uv4u\n"
                           "tff4A9Bm3V9tLx9S6djq31w31Gl7OQhryodW28kc16t9TvO1BzcV3HjRPwpe701X\n"
                           "oEEZdnZWANkkpR/m/pfgdmGPU66S2sXMHgsliViQWpDCYeehrvFRHEdR9NV+XJfC\n"
                           "QMUk26jPTIVTLfXmmwU0u8vUkpR7LQKkwwIBAg==\n"
                           "-----END DH PARAMETERS-----\n";*/


    ctx.set_options(boost::asio::ssl::context::default_workarounds |
                    boost::asio::ssl::context::no_sslv2);
                    //boost::asio::ssl::context::single_dh_use);

    ctx.use_certificate_chain(boost::asio::buffer(cert.data(), cert.size()));

    ctx.use_private_key(boost::asio::buffer(key.data(), key.size()),
                        boost::asio::ssl::context::file_format::pem);

    //ctx.use_tmp_dh(boost::asio::buffer(dh.data(), dh.size()));

    return true;
}

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
        if(loadCertificates(m_ctx, error) == false)
        {
            LOG_ERROR(error);
            return;
        }

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
