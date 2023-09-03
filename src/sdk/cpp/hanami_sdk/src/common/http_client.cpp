/**
 * @file        request.cpp
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

#include <common/http_client.h>

#include <hanami_json/json_item.h>

namespace Hanami
{

Hanami::HanamiRequest* HanamiRequest::m_instance = nullptr;

/**
 * @brief constructor
 */
HanamiRequest::HanamiRequest() {}

const std::string&
HanamiRequest::getHost() const
{
    return m_host;
}

const std::string&
HanamiRequest::getPort() const
{
    return m_port;
}

/**
 * @brief HanamiRequest::token
 * @return
 */
const std::string&
HanamiRequest::getToken() const
{
    return m_token;
}

void
HanamiRequest::updateToken(const std::string &newToken)
{
    m_token = newToken;
}

/**
 * @brief static methode to get instance of the interface
 *
 * @return pointer to the static instance
 */
HanamiRequest*
HanamiRequest::getInstance()
{
    if(m_instance == nullptr) {
        m_instance = new HanamiRequest();
    }

    return m_instance;
}

/**
 * @brief destructor
 */
HanamiRequest::~HanamiRequest()
{
}

/**
 * @brief init request-object
 *
 * @param host target-host-address
 * @param port port of the server
 * @param user user-name for token-request
 * @param password password for token-request
 * @param cacert cacert for connection-validation
 *
 * @return false, if host or port ar missing in variables and venv, else true
 */
bool
HanamiRequest::init(const std::string &host,
                    const std::string &port,
                    const std::string &user,
                    const std::string &password,
                    const std::string &cacert)
{
    m_host = host;
    m_port = port;
    m_cacert = cacert;
    m_userId = user;
    m_password = password;

    // get host-address
    if(m_host == ""
        && getEnvVar(m_host, "HANAMI_ADDRESS") == false)
    {
        return false;
    }

    // get target-port
    if(m_port == ""
        && getEnvVar(m_port, "HANAMI_PORT") == false)
    {
        return false;
    }

    // ge ca-cert
    if(m_cacert == "") {
        getEnvVar(m_cacert, "HANAMI_CACERT");
    }

    // get token if there already one exist
    getEnvVar(m_token, "HANAMI_TOKEN");

    return true;
}

/**
 * @brief Request::sendGetRequest
 *
 * @param response reference for response-output
 * @param path path to call
 * @param vars variables as string for the request-path
 * @param error reference for error-output
 *
 * @return false, if something went wrong while sending or token-request failed, else true
 */
bool
HanamiRequest::sendGetRequest(std::string &response,
                              const std::string &path,
                              const std::string &vars,
                              Hanami::ErrorContainer &error)
{
    return makeRequest(response, http::verb::get, path, vars, "", error);
}

/**
 * @brief Request::sendPostRequest
 *
 * @param response reference for response-output
 * @param path path to call
 * @param vars variables as string for the request-path
 * @param body json-body as string
 * @param error reference for error-output
 *
 * @return false, if something went wrong while sending or token-request failed, else true
 */
bool
HanamiRequest::sendPostRequest(std::string &response,
                               const std::string &path,
                               const std::string &vars,
                               const std::string &body,
                               Hanami::ErrorContainer &error)
{
    return makeRequest(response, http::verb::post, path, vars, body, error);
}

/**
 * @brief Request::sendPutRequest
 *
 * @param response reference for response-output
 * @param path path to call
 * @param vars variables as string for the request-path
 * @param body json-body as string
 * @param error reference for error-output
 *
 * @return false, if something went wrong while sending or token-request failed, else true
 */
bool
HanamiRequest::sendPutRequest(std::string &response,
                              const std::string &path,
                              const std::string &vars,
                              const std::string &body,
                              Hanami::ErrorContainer &error)
{
    return makeRequest(response, http::verb::put, path, vars, body, error);
}

/**
 * @brief Request::sendDeleteRequest
 *
 * @param response reference for response-output
 * @param path path to call
 * @param vars variables as string for the request-path
 * @param error reference for error-output
 *
 * @return false, if something went wrong while sending or token-request failed, else true
 */
bool
HanamiRequest::sendDeleteRequest(std::string &response,
                                 const std::string &path,
                                 const std::string &vars,
                                 Hanami::ErrorContainer &error)
{
    return makeRequest(response, http::verb::delete_, path, vars, "", error);
}

/**
 * @brief get content of an environment-variable
 *
 * @param content reference for output
 * @param key name of the environment-variable
 *
 * @return false, if varibale is not set, else true
 */
bool
HanamiRequest::getEnvVar(std::string &content,
                         const std::string &key) const
{
    const char* val = getenv(key.c_str());
    if(val == NULL) {
        return false;
    }

    content = std::string(val);
    return true;
}

/**
 * @brief Request::requestToken
 *
 * @param error reference for error-output
 *
 * @return false, if something failed, else true
 */
bool
HanamiRequest::requestToken(Hanami::ErrorContainer &error)
{
    // get user for access
    if(m_userId == ""
        && getEnvVar(m_userId, "HANAMI_USER_ID") == false)
    {
        error.addMeesage("Failed to request token, because no user-id was provided");
        LOG_ERROR(error);
        return false;
    }

    // get password for access
    if(m_password == ""
        && getEnvVar(m_password, "HANAMI_USER_PW") == false)
    {
        error.addMeesage("Failed to request token, because no password was provided");
        LOG_ERROR(error);
        return false;
    }

    // build request-path and body
    const std::string path = "/control/v1/token";
    const std::string jsonBody = "{\"id\":\""
                                 + m_userId
                                 + "\",\"password\":\""
                                 + m_password
                                 + "\"}";

    // make token-request
    std::string response;
    if(makeSingleRequest(response, http::verb::post, path, jsonBody, error) != 200)
    {
        error.addMeesage("Failed to request token");
        LOG_ERROR(error);
        return false;
    }

    // try to parse response
    Hanami::JsonItem item;
    if(item.parse(response, error) == false)
    {
        error.addMeesage("Failed to parse token-response");
        LOG_ERROR(error);
        return false;
    }

    // get token from parsed item
    m_token = item["token"].getString();
    if(m_token == "")
    {
        error.addMeesage("Can not find token in token-response");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief make a request against the backend
 *
 * @param response reference for response-output
 * @param type request-type
 * @param path path to call
 * @param vars variables as string for the request-path
 * @param jsonBody json-body as string
 * @param error reference for error-output
 *
 * @return false, if something went wrong while sending or token-request failed, else true
 */
bool
HanamiRequest::makeRequest(std::string &response,
                           const http::verb type,
                           const std::string &path,
                           const std::string &vars,
                           const std::string &jsonBody,
                           Hanami::ErrorContainer &error)
{
    // get token if necessary
    if(m_token == "")
    {
        if(requestToken(error) == false) {
            return false;
        }
    }

    // build real request-path with the ntoken
    std::string target = path;
    if(vars != "") {
        target.append("?" + vars);
    }

    // send request
    if(makeSingleRequest(response, type, target, jsonBody, error) != 200) {
        return false;
    }

    // handle expired token
    if(response == "Token is expired")
    {
        // request new token
        if(requestToken(error) == false) {
            return false;
        }

        // build new request-path with the new token
        target = path;
        if(m_token != "") {
            target += "?token=" + m_token;
        }
        if(vars != "") {
            target.append("&" + vars);
        }

        // try request again
        if(makeSingleRequest(response, type, target, jsonBody, error) != 200) {
            return false;
        }
    }

    return true;
}

/**
 * @brief make a generic request against the backend
 *
 * @param response reference for response-output
 * @param type type of the request
 * @param target target-path as string
 * @param jsonBody json-body as string
 * @param error reference for error-output
 *
 * @return false, if something went wrong while sending, else true
 */
uint16_t
HanamiRequest::makeSingleRequest(std::string &response,
                                 const http::verb type,
                                 const std::string &target,
                                 const std::string &jsonBody,
                                 Hanami::ErrorContainer &error)
{
    u_int16_t statusCode = 0;

    try
    {
        int version = 11;

        // init ssl
        ssl::context ctx(ssl::context::tlsv13_client);
        //load_root_certificates(ctx);
        //ctx.set_verify_mode(ssl::verify_peer);

        // These objects perform our I/O
        net::io_context ioc;
        tcp::resolver resolver(ioc);
        beast::ssl_stream<beast::tcp_stream> stream(ioc, ctx);

        // Set SNI Hostname (many hosts need this to handshake successfully)
        if(! SSL_set_tlsext_host_name(stream.native_handle(), m_host.c_str()))
        {
            beast::error_code ec{
                static_cast<int>(::ERR_get_error()),
                net::error::get_ssl_category()
            };
            throw beast::system_error{ec};
        }

        // init connection
        const auto results = resolver.resolve(m_host, m_port);
        beast::get_lowest_layer(stream).connect(results);
        stream.handshake(ssl::stream_base::client);

        LOG_DEBUG("send http-request to '" + target + "'");

        // create request
        http::request<http::string_body> req{type, target, version};
        req.set(http::field::host, m_host);
        req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);

        // add token
        if(m_token != "") {
            req.set("X-Auth-Token", m_token);
        }

        // add body
        if(jsonBody.size() > 0)
        {
           req.body() = jsonBody;
           req.set(http::field::content_type, "application/json");
           req.content_length(jsonBody.size());
           req.prepare_payload();
        }

        // run request
        http::write(stream, req);

        // receive response
        beast::flat_buffer buffer;
        http::response<http::string_body> res;
        http::read(stream, buffer, res);
        response = res.body().c_str();
        statusCode = res.result_int();
        if(statusCode != 200)
        {
            if(statusCode == 500) {
                response = "Internal error";
            }
            error.addMeesage("ERROR " + std::to_string(statusCode) + ": " + response);
        }

        // Gracefully close the stream
        beast::error_code ec;
        stream.shutdown(ec);
        if(ec == net::error::eof)
        {
            // Rationale:
            // http://stackoverflow.com/questions/25587403/
            //        boost-asio-ssl-async-shutdown-always-finishes-with-an-error
            ec = {};
        }
        if(ec)
        {
            error.addMeesage("failed to close http-connection");
            return statusCode;
        }
    }
    catch(std::exception const& e)
    {
        error.addMeesage("Error while making http-request: "
                         + std::string(e.what()));
        return statusCode;
    }

    return statusCode;
}

} // namespace Hanami
