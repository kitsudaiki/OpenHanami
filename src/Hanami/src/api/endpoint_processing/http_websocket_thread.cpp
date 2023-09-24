/**
 * @file        http_websocket_thread.cpp
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

#include "http_websocket_thread.h"

#include <hanami_root.h>
#include <api/websocket/file_upload.h>
#include <api/websocket/cluster_io.h>
#include <api/endpoint_processing/http_server.h>
#include <api/endpoint_processing/http_processing/http_processing.h>
#include <core/cluster/cluster.h>
#include <core/cluster/cluster_handler.h>

#include <hanami_common/threading/event.h>

using namespace Hanami;

/**
 * @brief constructor
 */
HttpWebsocketThread::HttpWebsocketThread(const std::string &threadName)
    : Thread(threadName) {}

/**
 * @brief HttpThread::run
 */
void
HttpWebsocketThread::run()
{
    while(m_abort == false)
    {
        tcp::socket* socket = HanamiRoot::httpServer->getSocket();
        if(socket != nullptr)
        {
            ErrorContainer error;

            if(handleSocket(socket, error) == false) {
                LOG_ERROR(error);
            }
            delete socket;
        }
        else
        {
            sleepThread(10000);
        }
    }
}

/**
 * @brief handle new incoming http-connection
 *
 * @param socket pointer to new socket to process
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HttpWebsocketThread::handleSocket(tcp::socket* socket,
                                  ErrorContainer &error)
{
    beast::ssl_stream<tcp::socket&> stream{*socket, std::ref(HanamiRoot::httpServer->m_ctx)};
    http::request<http::string_body> httpRequest;
    http::response<http::dynamic_body> httpResponse;
    bool processResult = true;

    // perform the SSL handshake
    beast::error_code ec;
    stream.handshake(boost::asio::ssl::stream_base::server, ec);
    if(ec.failed())
    {
        error.addMeesage("SSL-Handshake failed while receiving new http-connection");
        return false;
    }

    // read http-message
    if(readMessage(stream, httpRequest, error) == false)
    {
        error.addMeesage("Can read http-request");
        return false;
    }

    // check if request belongs to a new websocket-request
    if(websocket::is_upgrade(httpRequest))
    {
        // initialize new websocket-session
        websocket::stream<beast::ssl_stream<tcp::socket&>> webSocket(std::move(stream));
        m_webSocket = &webSocket;        
        if(initWebsocket(httpRequest) == false)
        {
            error.addMeesage("Can not init websocket.");
            return false;
        }

        runWebsocket();
        m_uuid = "";
    }
    else
    {
        // process request
        processResult = processRequest(httpRequest, httpResponse, error);
        if(processResult == false)
        {
            error.addMeesage("Failed to process http-request.");
            // IMPORANT: no return false here, because the reponse should retunred anyway
        }
        if(sendResponse(stream, httpResponse, error) == false)
        {
            error.addMeesage("Can not send http-response.");
            return false;
        }

        // close socket gain
        beast::error_code ec;
        stream.shutdown(ec);

        if(ec)
        {
            error.addMeesage("Error while closing http-stream: " + ec.message());
            return false;
        }
    }

    return processResult;
}

/**
 * @brief handle message coming over the http-socket
 *
 * @param stream incoming stream
 * @param httpRequest incoming request over the stream
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HttpWebsocketThread::readMessage(beast::ssl_stream<tcp::socket&> &stream,
                                 http::request<http::string_body> &httpRequest,
                                 ErrorContainer &error)
{
    beast::error_code ec;
    beast::flat_buffer buffer;
    http::read(stream, buffer, httpRequest, ec);

    if(ec == http::error::end_of_stream) {
         return true;
    }

    if(ec)
    {
        error.addMeesage("Error while reading http-message: '" + ec.message() + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief send message over the http-socket
 *
 * @param stream outgoing stream
 * @param httpResponse response to send over the stream
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HttpWebsocketThread::sendResponse(beast::ssl_stream<tcp::socket&> &stream,
                                  http::response<http::dynamic_body> &httpResponse,
                                  ErrorContainer &error)
{
    beast::error_code ec;
    httpResponse.content_length(httpResponse.body().size());
    http::write(stream, httpResponse, ec);

    if(ec)
    {
        error.addMeesage("Error while writing http-message: '" + ec.message() + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief initialize websocket-connection
 *
 * @param httpRequest initial http-request to init the websocket
 *
 * @return true, if successful, else false
 */
bool
HttpWebsocketThread::initWebsocket(http::request<http::string_body> &httpRequest)
{
    try
    {
        // Set a decorator to change the Server of the handshake
        m_webSocket->set_option(websocket::stream_base::decorator(
            [](websocket::response_type& res)
            {
                res.set(http::field::server,
                    std::string(BOOST_BEAST_VERSION_STRING) +
                        " torii-websocket-ssl");
            }));

        // Accept the websocket handshake
        m_webSocket->accept(std::move(httpRequest));
    }
    catch(const beast::system_error& se)
    {
        if(se.code() == websocket::error::closed)
        {
            LOG_INFO("Close websocket1");
        }
        else
        {
            ErrorContainer error;
            error.addMeesage("Error while receiving data over websocket with message: "
                             + se.code().message());
            LOG_ERROR(error);
            return false;
        }
    }
    catch(const std::exception& e)
    {
        ErrorContainer error;
        error.addMeesage("Error while receiving data over websocket with message: "
                         + std::string(e.what()));
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief send data over web-socket
 *
 * @param data pointer to buffer with data to send
 * @param dataSize number of bytes to send
 *
 * @return true, if successful, else false
 */
bool
HttpWebsocketThread::sendData(const void* data,
                              const uint64_t dataSize)
{
    if(m_abort) {
        return false;
    }

    try
    {
        while(m_waitForInput == true
              && m_websocketClosed == false)
        {
            usleep(1000);
        }

        if(m_websocketClosed) {
            return false;
        }

        beast::flat_buffer buffer;
        m_webSocket->binary(true);
        m_webSocket->write(net::buffer(data, dataSize));
        m_waitForInput = true;

        return true;
    }
    catch(const beast::system_error& se)
    {
        if(se.code() == websocket::error::closed)
        {
            LOG_INFO("Close websocket2");
        }
        else
        {
            ErrorContainer error;
            error.addMeesage("Error while sending data over websocket with message: "
                             + se.code().message());
            LOG_ERROR(error);
        }
    }
    catch(const std::exception& e)
    {
        ErrorContainer error;
        error.addMeesage("Error while sending data over websocket with message: "
                         + std::string(e.what()));
        LOG_ERROR(error);
    }

    m_waitForInput = true;

    return false;
}

/**
 * @brief run the initial process to forward a websocket-connection to the backend
 *
 * @param message initial message to enable message-forwarding
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HttpWebsocketThread::processInitialMessage(const std::string &message,
                                           ErrorContainer &error)
{
    // precehck if already init
    if(m_clientInit)
    {
        error.addMeesage("Websocket alread initialized and can not be initialized again.");
        LOG_ERROR(error);
        return false;
    }

    // parse incoming initializing message
    json content;
    try {
        content = json::parse(message);
    } catch(const json::parse_error& ex) {
        error.addMeesage("Parsing of initial websocket-message failed");
        error.addMeesage("json-parser error: " + std::string(ex.what()));
        LOG_ERROR(error);
        return false;
    }

    RequestMessage requestMsg;
    ResponseMessage responseMsg;

    requestMsg.id = "v1/foreward";
    requestMsg.httpType = HttpRequestType::GET_TYPE;
    m_target = content["target"];

    // check authentication
    json tokenData;
    if(checkPermission(tokenData,
                       content["token"],
                       requestMsg,
                       responseMsg,
                       error) == false)
    {
        error.addMeesage("Request to misaki for token-check failed");
        LOG_ERROR(error);
        return false;
    }

    // handle failed authentication
    if(responseMsg.type == UNAUTHORIZED_RTYPE
            || responseMsg.success == false)
    {
        error.addMeesage("Permission-check for token over websocket failed");
        LOG_ERROR(error);
        return false;
    }

    // forward connection to shiori or hanami
    if(m_target == "kyouko")
    {
        const std::string getClusterUuid = content["uuid"];
        m_targetCluster = ClusterHandler::getInstance()->getCluster(getClusterUuid);
        m_targetCluster->msgClient = this;
        m_clientInit = true;
        return true;
    }
    else if(m_target == "shiori")
    {
        m_clientInit = true;
        return true;
    }

    error.addMeesage("Session-forwarding to target '" + m_target + "' is not allowed");
    LOG_ERROR(error);

    return false;
}

/**
 * @brief close temporary client, if exist
 *
 * @param error reference for error-output
 */
void
HttpWebsocketThread::closeClient(ErrorContainer &)
{
    m_clientInit = false;
    if(m_targetCluster != nullptr) {
        m_targetCluster->msgClient = nullptr;
    }
    m_targetCluster = nullptr;
}

/**
 * @brief run the websocket
 */
void
HttpWebsocketThread::runWebsocket()
{
    ErrorContainer error;
    m_websocketClosed = false;

    try
    {
        while(m_abort == false)
        {
            // read message from socket
            beast::flat_buffer buffer;
            while(m_waitForInput == false
                  && m_websocketClosed == false)
            {
                usleep(1000);
            }

            if(m_websocketClosed) {
                break;
            }

            m_webSocket->read(buffer);
            m_waitForInput = false;

            //m_webSocket.text(m_webSocket.got_text());
            if(m_clientInit == false)
            {
                const std::string msg(static_cast<const char*>(buffer.data().data()),
                                      buffer.data().size());
                m_uuid = generateUuid().toString();

                LOG_DEBUG("got initial websocket-message: '" + msg + "'");
                bool success = true;
                if(processInitialMessage(msg, error) == false)
                {
                    success = false;
                    error.addMeesage("Failed initializing of websocket-forwarding");
                    LOG_ERROR(error);
                    error = ErrorContainer();
                }

                // build response-message
                json response = json::object();
                response["success"] = success;
                if(success) {
                    response["uuid"] = m_uuid;
                }

                const std::string responseMsg = response.dump();
                m_webSocket->binary(true);
                m_webSocket->write(net::buffer(responseMsg, responseMsg.size()));
                m_waitForInput = true;
            }
            else
            {
                if(m_target == "kyouko")
                {
                    recvClusterInputMessage(m_targetCluster,
                                            buffer.data().data(),
                                            buffer.data().size());
                }
                else if(m_target == "shiori")
                {
                    recvFileUploadPackage(buffer.data().data(),
                                          buffer.data().size());
                    m_waitForInput = true;
                }
            }
        }
    }
    catch(const beast::system_error& se)
    {
        if(se.code() == websocket::error::closed)
        {
            LOG_INFO("Close websocket3");
        }
        else
        {
            ErrorContainer error;
            error.addMeesage("Error while receiving data over websocket with message: "
                             + se.code().message());
        }
    }
    catch(const std::exception& e)
    {
        ErrorContainer error;
        error.addMeesage("Error while receiving data over websocket with message: "
                         + std::string(e.what()));
    }

    m_websocketClosed = true;
    closeClient(error);
    LOG_ERROR(error);
}
