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

#ifndef HANAMI_STRUCTS_H
#define HANAMI_STRUCTS_H

#include <stdint.h>
#include <string>
#include <uuid/uuid.h>

#include <common/defines.h>
#include <common/enums.h>

#include <hanami_policies/policy.h>

#define UNINTI_POINT_32 0x0FFFFFFF

struct NextSides
{
    uint8_t sides[5];
};

struct ResponseMessage
{
    bool success = false;
    HttpResponseTypes type = NO_CONTENT_RTYPE;
    std::string responseContent = "";
};

struct RequestMessage
{
    Hanami::HttpRequestType httpType = Hanami::GET_TYPE;
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

    UserContext(const json &inputContext)
    {
        userId = inputContext["id"];
        projectId = inputContext["project_id"];
        isAdmin = inputContext["is_admin"];
        isProjectAdmin = inputContext["is_project_admin"];
        if(inputContext.contains("token")) {
            token = inputContext["token"];
        }
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

#endif // HANAMI_STRUCTS_H
