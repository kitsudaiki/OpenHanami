/**
 * @file        file_send.h
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

#ifndef TORIIGATEWAY_FILE_SEND_H
#define TORIIGATEWAY_FILE_SEND_H

#include <boost/beast/http.hpp>
#include <boost/beast/core.hpp>

#include <string>
#include <filesystem>

#include <hanami_common/logger.h>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>

const std::string getResponseType(const std::string &ext);
bool sendFileFromLocalLocation(http::response<http::dynamic_body> &response,
                               const std::string &dir,
                               const std::string &relativePath,
                               Hanami::ErrorContainer &error);
bool processClientRequest(http::response<http::dynamic_body> &response,
                          const std::string &path,
                          Hanami::ErrorContainer &error);

#endif // TORIIGATEWAY_FILE_SEND_H
