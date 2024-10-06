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

#include <api/endpoint_processing/auth_check.h>
#include <api/endpoint_processing/blossom.h>
#include <api/endpoint_processing/http_processing/http_processing.h>
#include <api/endpoint_processing/http_server.h>
#include <api/http/endpoint_init.h>
#include <api/websocket/cluster_io.h>
#include <api/websocket/file_upload.h>
#include <core/cluster/cluster.h>
#include <core/cluster/cluster_handler.h>
#include <core/temp_file_handler.h>
#include <database/cluster_table.h>
#include <database/tempfile_table.h>
#include <hanami_common/threading/event.h>
#include <hanami_root.h>

using namespace Hanami;

/**
 * @brief constructor
 */
HttpWebsocketThread::HttpWebsocketThread(const std::string& threadName) : Thread(threadName) {}

/**
 * @brief HttpThread::run
 */
void
HttpWebsocketThread::run()
{
    while (m_abort == false) {
        tcp::socket* socket = HanamiRoot::httpServer->getSocket();
        if (socket != nullptr) {
            handleSocket(socket);
            delete socket;
        }
        else {
            sleepThread(10000);
        }
    }
}

/**
 * @brief handle new incoming http-connection
 *
 * @param socket pointer to new socket to process
 *
 * @return true, if successful, else false
 */
bool
HttpWebsocketThread::handleSocket(tcp::socket* socket)
{
    ErrorContainer error;
    http::request<http::string_body> httpRequest;
    http::response<http::dynamic_body> httpResponse;
    bool processResult = true;

    // read http-message
    if (readMessage(*socket, httpRequest, error) == false) {
        error.addMessage("Can read http-request");
        return false;
    }

    // check if request belongs to a new websocket-request
    if (websocket::is_upgrade(httpRequest)) {
        // initialize new websocket-session
        websocket::stream<tcp::socket&> webSocket(*socket);
        m_webSocket = &webSocket;
        if (initWebsocket(httpRequest) == false) {
            error.addMessage("Can not init websocket.");
            return false;
        }

        runWebsocket();
        m_webSocket = nullptr;
        m_uuid = "";
    }
    else {
        // process request
        processResult = HanamiRoot::httpServer->httpProcessing->processRequest(
            httpRequest, httpResponse, error);
        if (processResult == false) {
            LOG_DEBUG("Failed to process http-request.");
        }
        if (sendResponse(*socket, httpResponse, error) == false) {
            error.addMessage("Can not send http-response.");
            return false;
        }

        // close socket gain
        socket->close();
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
HttpWebsocketThread::readMessage(tcp::socket& stream,
                                 http::request<http::string_body>& httpRequest,
                                 ErrorContainer& error)
{
    beast::error_code ec;
    beast::flat_buffer buffer;
    http::read(stream, buffer, httpRequest, ec);

    if (ec == http::error::end_of_stream) {
        return true;
    }

    if (ec) {
        error.addMessage("Error while reading http-message: '" + ec.message() + "'");
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
HttpWebsocketThread::sendResponse(tcp::socket& socket,
                                  http::response<http::dynamic_body>& httpResponse,
                                  ErrorContainer& error)
{
    beast::error_code ec;
    httpResponse.content_length(httpResponse.body().size());
    http::write(socket, httpResponse, ec);

    if (ec) {
        error.addMessage("Error while writing http-message: '" + ec.message() + "'");
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
HttpWebsocketThread::initWebsocket(http::request<http::string_body>& httpRequest)
{
    try {
        // Set a decorator to change the Server of the handshake
        m_webSocket->set_option(websocket::stream_base::decorator(
            [](websocket::response_type& res)
            {
                res.set(http::field::server,
                        std::string(BOOST_BEAST_VERSION_STRING) + " torii-websocket-ssl");
            }));

        // Accept the websocket handshake
        m_webSocket->accept(std::move(httpRequest));
    }
    catch (const beast::system_error& se) {
        if (se.code() == websocket::error::closed) {
            LOG_INFO("Close websocket1");
        }
        else {
            ErrorContainer error;
            error.addMessage("Error while receiving data over websocket with message: "
                             + se.code().message());
            LOG_ERROR(error);
            return false;
        }
    }
    catch (const std::exception& e) {
        ErrorContainer error;
        error.addMessage("Error while receiving data over websocket with message: "
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
HttpWebsocketThread::sendData(const void* data, const uint64_t dataSize)
{
    if (m_abort) {
        return false;
    }

    try {
        while (m_waitForInput == true && m_websocketClosed == false) {
            usleep(1000);
        }

        if (m_websocketClosed) {
            return false;
        }

        beast::flat_buffer buffer;
        m_webSocket->binary(true);
        m_webSocket->write(net::buffer(data, dataSize));
        m_waitForInput = true;

        return true;
    }
    catch (const beast::system_error& se) {
        if (se.code() == websocket::error::closed) {
            LOG_INFO("Close websocket2");
        }
        else {
            ErrorContainer error;
            error.addMessage("Error while sending data over websocket with message: "
                             + se.code().message());
            LOG_ERROR(error);
        }
    }
    catch (const std::exception& e) {
        ErrorContainer error;
        error.addMessage("Error while sending data over websocket with message: "
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
HttpWebsocketThread::processInitialMessage(const std::string& message, std::string& errorMessage)
{
    ErrorContainer error;

    // precehck if already init
    if (m_clientInit) {
        error.addMessage("Websocket alread initialized and can not be initialized again.");
        LOG_ERROR(error);
        return false;
    }

    // parse incoming initializing message
    json content;
    try {
        content = json::parse(message);
    }
    catch (const json::parse_error& ex) {
        errorMessage = "json-parser error: " + std::string(ex.what());
        return false;
    }

    // check authentication
    BlossomStatus status;
    json tokenData = json::object();
    if (validateToken(tokenData, content["token"], "", HttpRequestType::GET_TYPE, errorMessage)
        == false)
    {
        errorMessage = "Token invalid";
        return false;
    }

    const Hanami::UserContext userContext = convertContext(tokenData);

    // forward connection to shiori or hanami
    if (content["target"] == "cluster") {
        const std::string clusterUuid = content["uuid"];

        // check if uuid exist in context of the user and project
        json clusterData = json::object();
        const ReturnStatus ret = ClusterTable::getInstance()->getCluster(
            clusterData, clusterUuid, userContext, false, error);
        if (ret == INVALID_INPUT) {
            errorMessage = "Cluster with UUID '" + clusterUuid + "' not found";
            return false;
        }
        if (ret == ERROR) {
            errorMessage = "Internal error.";
            return false;
        }

        // ini local socket
        m_targetCluster = ClusterHandler::getInstance()->getCluster(clusterUuid);
        m_targetCluster->msgClient = this;
        m_clientInit = true;
        m_target = content["target"];

        return true;
    }
    else if (content["target"] == "file_upload") {
        const std::string fileUuid = content["uuid"];

        // check if uuid exist in context of the user and project
        json tempfileData = json::object();
        const ReturnStatus ret = TempfileTable::getInstance()->getTempfile(
            tempfileData, fileUuid, userContext, false, error);
        if (ret == INVALID_INPUT) {
            errorMessage = "Tempfile with UUID '" + fileUuid + "' not found";
            return false;
        }
        if (ret == ERROR) {
            errorMessage = "Internal error";
            return false;
        }

        // ini local socket
        m_fileHandle = TempFileHandler::getInstance()->getFileHandle(fileUuid, userContext);
        if (m_fileHandle == nullptr) {
            errorMessage = "Tempfile with UUID '" + fileUuid + "' not found";
            error.addMessage("Tempfile with UUID '" + fileUuid + "'exist in database, "
                             "but not in file-handler");
            LOG_ERROR(error, userContext.userId);
            return false;
        }
        m_fileHandle->lock = true;
        m_fileHandle->timeoutCounter = 0;
        m_clientInit = true;
        m_target = content["target"];

        return true;
    }

    return false;
}

/**
 * @brief close temporary client, if exist
 */
void
HttpWebsocketThread::closeClient()
{
    m_clientInit = false;
    if (m_targetCluster != nullptr) {
        m_targetCluster->msgClient = nullptr;
    }
    m_targetCluster = nullptr;

    if (m_fileHandle != nullptr) {
        m_fileHandle->lock = false;
    }
    m_fileHandle = nullptr;
}

/**
 * @brief run the websocket
 */
void
HttpWebsocketThread::runWebsocket()
{
    m_websocketClosed = false;

    try {
        while (m_abort == false) {
            // read message from socket
            beast::flat_buffer buffer;
            while (m_waitForInput == false && m_websocketClosed == false) {
                usleep(1000);
            }

            if (m_websocketClosed) {
                break;
            }

            m_webSocket->read(buffer);
            m_waitForInput = false;

            // m_webSocket.text(m_webSocket.got_text());
            if (m_clientInit == false) {
                const std::string msg(static_cast<const char*>(buffer.data().data()),
                                      buffer.data().size());
                m_uuid = generateUuid().toString();

                LOG_DEBUG("got initial websocket-message: '" + msg + "'");
                std::string errorMessage = "";
                const bool success = processInitialMessage(msg, errorMessage);

                // build response-message
                json response = json::object();
                response["success"] = success;
                if (success) {
                    response["uuid"] = m_uuid;
                }
                else {
                    response["error"] = errorMessage;
                }

                const std::string responseMsg = response.dump();
                m_webSocket->binary(true);
                m_webSocket->write(net::buffer(responseMsg, responseMsg.size()));
                m_waitForInput = true;
            }
            else {
                if (m_target == "cluster") {
                    recvClusterInputMessage(
                        m_targetCluster, buffer.data().data(), buffer.data().size());
                }
                else if (m_target == "file_upload") {
                    std::string errorMessage;
                    const bool ret = recvFileUploadPackage(
                        m_fileHandle, buffer.data().data(), buffer.data().size(), errorMessage);
                    sendFileUploadResponse(this, ret, errorMessage);
                }
            }
        }
    }
    catch (const beast::system_error& se) {
        if (se.code() == websocket::error::closed) {
            LOG_DEBUG("Close websocket3");
        }
        else {
            ErrorContainer error;
            error.addMessage("Error while receiving data over websocket with message: "
                             + se.code().message());
            LOG_ERROR(error);
        }
    }
    catch (const std::exception& e) {
        ErrorContainer error;
        error.addMessage("Error while receiving data over websocket with message: "
                         + std::string(e.what()));
        LOG_ERROR(error);
    }

    m_websocketClosed = true;
    closeClient();
}
