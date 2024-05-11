/**
 * @file        functions.h
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

#ifndef HANAMI_FUNCTIONS_H
#define HANAMI_FUNCTIONS_H

#include <common/defines.h>
#include <common/structs.h>
#include <core/cluster/objects.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

/**
 * @brief get the current datetime of the system
 *
 * @return datetime as string
 */
inline const std::string
getDatetime()
{
    const time_t now = time(nullptr);
    tm* ltm = localtime(&now);

    const std::string datatime
        = std::to_string(1900 + ltm->tm_year) + "-" + std::to_string(1 + ltm->tm_mon) + "-"
          + std::to_string(ltm->tm_mday) + " " + std::to_string(ltm->tm_hour) + ":"
          + std::to_string(ltm->tm_min) + ":" + std::to_string(ltm->tm_sec);

    return datatime;
}

/**
 * @brief convert chrono-timestamp into a string in UTC time
 *
 * @param time chrono-timeshamp, which should be converted
 * @param format format to convert into
 *
 * @return string with the converted timestamp
 */
inline const std::string
serializeTimePoint(const std::chrono::high_resolution_clock::time_point& time,
                   const std::string& format = "%Y-%m-%d %H:%M:%S")
{
    std::time_t tt = std::chrono::system_clock::to_time_t(time);
    std::tm tm = *std::gmtime(&tt);
    std::stringstream ss;
    ss << std::put_time(&tm, format.c_str());
    return ss.str();
}

/**
 * @brief check if an id is an uuid
 *
 * @param id id to check
 *
 * @return true, if id is an uuid, else false
 */
inline bool
isUuid(const std::string& id)
{
    const std::regex uuidRegex(UUID_REGEX);
    return regex_match(id, uuidRegex);
}

/**
 * @brief generate a new uuid with external library
 *
 * @return new uuid
 */
inline const kuuid
generateUuid()
{
    uuid_t binaryUuid;
    kuuid result;

    uuid_generate_random(binaryUuid);
    uuid_unparse_lower(binaryUuid, result.uuid);

    return result;
}

/**
 * @brief function for generating random-values
 *        coming from this website:
 *            https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
 *
 * @param input seed for random value
 *
 * @return random value
 */
inline uint32_t
pcg_hash(const uint32_t input)
{
    const uint32_t state = input * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

#endif  // HANAMI_FUNCTIONS_H
