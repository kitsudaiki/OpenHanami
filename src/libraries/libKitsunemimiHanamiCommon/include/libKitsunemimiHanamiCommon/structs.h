/**
 * @file        structs.h
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

#ifndef KITSUNEMIMI_HANAMI_COMMON_STRUCTS_H
#define KITSUNEMIMI_HANAMI_COMMON_STRUCTS_H

#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiHanamiCommon/defines.h>
#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi::Hanami
{

struct ResponseMessage
{
    bool success = false;
    HttpResponseTypes type = NO_CONTENT_RTYPE;
    std::string responseContent = "";
};

struct RequestMessage
{
    HttpRequestType httpType = GET_TYPE;
    std::string id = "";
    std::string inputValues = "{}";
};

struct UserContext
{
    std::string userId = "";
    std::string projectId = "";
    bool isAdmin = false;
    bool isProjectAdmin = false;
    std::string token = "";

    UserContext() {}

    UserContext(const DataMap &inputContext)
    {
        userId = inputContext.getStringByKey("id");
        projectId = inputContext.getStringByKey("project_id");
        isAdmin = inputContext.getBoolByKey("is_admin");
        isProjectAdmin = inputContext.getBoolByKey("is_project_admin");
        token = inputContext.getStringByKey("token");
    }
};

struct Position
{
    uint32_t x = UNINTI_POINT_32;
    uint32_t y = UNINTI_POINT_32;
    uint32_t z = UNINTI_POINT_32;
    uint32_t w = UNINTI_POINT_32;

    Position() {}

    Position(const Position &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
    }

    Position& operator=(const Position &other)
    {
        if(this != &other)
        {
            x = other.x;
            y = other.y;
            z = other.z;
        }

        return *this;
    }

    bool operator==(const Position &other)
    {
        return(this->x == other.x
               && this->y == other.y
               && this->z == other.z);
    }

    bool isValid() const
    {
        return(x != UNINTI_POINT_32
               && y != UNINTI_POINT_32
               && z != UNINTI_POINT_32);
    }
};

struct EndpointEntry
{
    SakuraObjectType type = BLOSSOM_TYPE;
    std::string group = "-";
    std::string name = "";
};


struct BlossomStatus
{
    uint64_t statusCode = 0;
    std::string errorMessage = "";
};

}

#endif // KITSUNEMIMI_HANAMI_COMMON_STRUCTS_H
