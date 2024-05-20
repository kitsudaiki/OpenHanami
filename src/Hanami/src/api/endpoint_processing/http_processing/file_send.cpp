/**
 * @file        file_send.cpp
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

#include "file_send.h"

#include <hanami_common/files/text_file.h>
#include <hanami_common/functions/string_functions.h>
#include <hanami_common/logger.h>
#include <hanami_config/config_handler.h>

/**
 * @brief get response-type for the requested file
 *
 * @param ext file-extension
 *
 * @return response-type
 */
const std::string
getResponseType(const std::string& ext)
{
    if (beast::iequals(ext, ".htm")) {
        return "text/html";
    }
    if (beast::iequals(ext, ".html")) {
        return "text/html";
    }
    if (beast::iequals(ext, ".php")) {
        return "text/html";
    }
    if (beast::iequals(ext, ".css")) {
        return "text/css";
    }
    if (beast::iequals(ext, ".txt")) {
        return "text/plain";
    }
    if (beast::iequals(ext, ".js")) {
        return "application/javascript";
    }
    if (beast::iequals(ext, ".json")) {
        return "application/json";
    }
    if (beast::iequals(ext, ".xml")) {
        return "application/xml";
    }
    if (beast::iequals(ext, ".swf")) {
        return "application/x-shockwave-flash";
    }
    if (beast::iequals(ext, ".flv")) {
        return "video/x-flv";
    }
    if (beast::iequals(ext, ".png")) {
        return "image/png";
    }
    if (beast::iequals(ext, ".jpe")) {
        return "image/jpeg";
    }
    if (beast::iequals(ext, ".jpeg")) {
        return "image/jpeg";
    }
    if (beast::iequals(ext, ".jpg")) {
        return "image/jpeg";
    }
    if (beast::iequals(ext, ".gif")) {
        return "image/gif";
    }
    if (beast::iequals(ext, ".bmp")) {
        return "image/bmp";
    }
    if (beast::iequals(ext, ".ico")) {
        return "image/vnd.microsoft.icon";
    }
    if (beast::iequals(ext, ".tiff")) {
        return "image/tiff";
    }
    if (beast::iequals(ext, ".tif")) {
        return "image/tiff";
    }
    if (beast::iequals(ext, ".svg")) {
        return "image/svg+xml";
    }
    if (beast::iequals(ext, ".svgz")) {
        return "image/svg+xml";
    }

    return "application/text";
}

/**
 * @brief send file, which was requested
 *
 * @return true, if successful, else false
 */
bool
sendFileFromLocalLocation(http::response<http::dynamic_body>& response,
                          const std::string& dir,
                          const std::string& relativePath,
                          Hanami::ErrorContainer& error)
{
    // create file-path
    std::string path = dir;
    if (relativePath == "/" || relativePath == "") {
        path += "/index.html";
    }
    else {
        path += relativePath;
    }

    LOG_DEBUG("load file " + path);

    // set response-type based on file-type
    std::filesystem::path pathObj(path);
    const std::string extension = pathObj.extension().string();
    response.set(http::field::content_type, getResponseType(extension));

    // read file and send content back
    std::string fileContent = "";
    if (Hanami::readFile(fileContent, path, error)) {
        beast::ostream(response.body()) << fileContent;
        return true;
    }

    response.result(http::status::internal_server_error);
    response.set(http::field::content_type, "text/plain");

    return false;
}

/**
 * @brief process file-request
 *
 * @param path requested file-apth
 *
 * @return false, if file not found, else true
 */
bool
processClientRequest(http::response<http::dynamic_body>& response,
                     const std::string& path,
                     Hanami::ErrorContainer& error)
{
    bool success = false;
    const std::string fileLocation = GET_STRING_CONFIG("http", "dashboard_files", success);
    // TODO: check success-flag
    return sendFileFromLocalLocation(response, fileLocation, path, error);
}
