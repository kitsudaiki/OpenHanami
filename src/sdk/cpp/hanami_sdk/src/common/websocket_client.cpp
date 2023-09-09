/**
 * @file        websocket_client.cpp
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

#include <hanami_sdk/common/websocket_client.h>

namespace Hanami
{

/**
 * @brief constructor
 */
WebsocketClient::WebsocketClient() {}

/**
 * @brief destructor
 */
WebsocketClient::~WebsocketClient()
{
    m_websocket->close(websocket::close_code::normal);
    // a 'delete' on the m_websocket-pointer breaks the program,
    // because of bad programming in the websocket-class of the boost beast library
    // TODO: fix the memory-leak
}

/**
 * @brief initialize new websocket-connection to a torii
 *
 * @param socketUuid reference for the
 * @param token token to authenticate socket on the torii
 * @param target name of the target on server-side behind the torii
 * @param host address of the torii
 * @param port port where the server is listen on target-side
 * @param targetUuid uuid of the target-resource
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
WebsocketClient::initClient(std::string &socketUuid,
                            const std::string &token,
                            const std::string &target,
                            const std::string &host,
                            const std::string &port,
                            const std::string &targetUuid,
                            Hanami::ErrorContainer &error)
{
    try
    {
        // init ssl
        ssl::context ctx{ssl::context::tlsv13_client};
        /*if(loadCertificates(ctx) == false)
        {
            error.addMeesage("Failed to load certificates for creating Websocket-Client");
            LOG_ERROR(error);
            return false;
        }*/

        net::io_context ioc;
        tcp::resolver resolver{ioc};
        m_websocket = new websocket::stream<beast::ssl_stream<tcp::socket>>{ioc, ctx};

        // Look up the domain name
        const auto results = resolver.resolve(host, port);
        auto ep = net::connect(get_lowest_layer(*m_websocket), results);

        // Set SNI Hostname (many hosts need this to handshake successfully)
        if(! SSL_set_tlsext_host_name(m_websocket->next_layer().native_handle(), host.c_str()))
            throw beast::system_error(
                beast::error_code(
                    static_cast<int>(::ERR_get_error()),
                    net::error::get_ssl_category()),
                "Failed to set SNI Hostname");

        const std::string address = host + ':' + std::to_string(ep.port());

        m_websocket->next_layer().handshake(ssl::stream_base::client);
        m_websocket->set_option(websocket::stream_base::decorator(
            [](websocket::response_type& res)
            {
                res.set(http::field::server,
                    std::string(BOOST_BEAST_VERSION_STRING) +
                        " client-websocket-ssl");
            }));

        // Perform the websocket handshake
        m_websocket->handshake(address, "/");

        std::string initialMsg = "";
        initialMsg.append("{\"token\":\"");
        initialMsg.append(token);
        initialMsg.append("\",\"target\":\"");
        initialMsg.append(target);
        initialMsg.append("\",\"uuid\":\"");
        initialMsg.append(targetUuid);
        initialMsg.append("\"}");

        // Send the message
        m_websocket->binary(true);
        m_websocket->write(net::buffer(initialMsg, initialMsg.size()));

        // Read a message into our buffer
        beast::flat_buffer buffer;
        m_websocket->read(buffer);

        const std::string responseMsg(static_cast<const char*>(buffer.data().data()),
                                      buffer.data().size());

        // parse response
        json response = json::parse(responseMsg, nullptr, false);
        if(response.is_discarded())
        {
            error.addMeesage("Failed to parse response-message from Websocket-init");
            LOG_ERROR(error);
            return false;
        }

        socketUuid = response["uuid"];
        return response["success"];
    }
    catch(std::exception const& e)
    {
        const std::string msg(e.what());
        error.addMeesage("Error-Message while initilializing Websocket-Client: '" + msg + "'");
        LOG_ERROR(error);
        return false;
    }

    return false;
}

/**
 * @brief send data over websocket-client
 *
 * @param data pointer to data to send
 * @param dataSize number of bytes to send
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
WebsocketClient::sendMessage(const void* data,
                             const uint64_t dataSize,
                             Hanami::ErrorContainer &error)
{
    try
    {
        // Send the message
        m_websocket->binary(true);
        m_websocket->write(net::buffer(data, dataSize));
    }
    catch(const std::exception &e)
    {
        const std::string msg(e.what());
        error.addMeesage("Error-Message while send Websocket-Data: '" + msg + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief WebsocketClient::readMessage
 *
 * @param numberOfByes reference for output of number of read bytes
 * @param error reference for error-output
 *
 * @return nullptr if failed, else pointer
 */
uint8_t*
WebsocketClient::readMessage(uint64_t &numberOfByes,
                             Hanami::ErrorContainer &error)
{
    try
    {
        // Read a message into our buffer
        beast::flat_buffer buffer;
        m_websocket->read(buffer);

        numberOfByes = buffer.data().size();
        if(numberOfByes == 0) {
            return nullptr;
        }
        uint8_t* data = new uint8_t[numberOfByes];
        memcpy(data, buffer.data().data(), numberOfByes);

        return data;
    }
    catch(const std::exception &e)
    {
        numberOfByes = 0;
        const std::string msg(e.what());
        error.addMeesage("Error-Message while read Websocket-Data: '" + msg + "'");
        LOG_ERROR(error);
        return nullptr;
    }

    numberOfByes = 0;
    return nullptr;
}

/**
 * @brief load ssl-certificates for ssl-encryption of websocket  (not used at the moment)
 *
 * @param ctx reference to ssl-context
 *
 * @return true, if successful, else false
 */
bool
WebsocketClient::loadCertificates(boost::asio::ssl::context& ctx)
{
    // TODO: use this functions to load specific certificates from file
    std::string errorMessage = "";
    std::string cert = "...";
    std::string key = "...";

    const std::string dh = "...";


    ctx.set_options(boost::asio::ssl::context::default_workarounds |
                    boost::asio::ssl::context::no_sslv2 |
                    boost::asio::ssl::context::single_dh_use);

    ctx.use_certificate_chain(boost::asio::buffer(cert.data(), cert.size()));

    ctx.use_private_key(boost::asio::buffer(key.data(), key.size()),
                        boost::asio::ssl::context::file_format::pem);

    ctx.use_tmp_dh(boost::asio::buffer(dh.data(), dh.size()));

    return true;
}

}  // namespace Hanami
