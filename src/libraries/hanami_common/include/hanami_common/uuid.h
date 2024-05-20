/**
 * @file        uuid.h
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

#ifndef HANAMI_UUID_H
#define HANAMI_UUID_H

#include <stdint.h>
#include <uuid/uuid.h>

#include <regex>
#include <string>

#include "defines.h"

struct UUID {
    char uuid[UUID_STR_LEN];
    uint8_t padding[3];

    const std::string toString() const { return std::string(uuid, UUID_STR_LEN - 1); }
};
static_assert(sizeof(UUID) == 40);

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
inline const UUID
generateUuid()
{
    uuid_t binaryUuid;
    UUID result;

    uuid_generate_random(binaryUuid);
    uuid_unparse_lower(binaryUuid, result.uuid);

    return result;
}

#endif  // HANAMI_UUID_H
