/**
 * @file        user.h
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

#ifndef HANAMISDK_USER_H
#define HANAMISDK_USER_H

#include <hanami_common/logger.h>

namespace Hanami
{

bool createUser(std::string &result,
                const std::string &userId,
                const std::string &userName,
                const std::string &password,
                const bool isAdmin,
                Hanami::ErrorContainer &error);

bool getUser(std::string &result, const std::string &userId, Hanami::ErrorContainer &error);

bool listUser(std::string &result, Hanami::ErrorContainer &error);

bool deleteUser(std::string &result, const std::string &userId, Hanami::ErrorContainer &error);

bool addProjectToUser(std::string &result,
                      const std::string &userId,
                      const std::string &projectId,
                      const std::string &role,
                      const bool isProjectAdmin,
                      Hanami::ErrorContainer &error);

bool removeProjectFromUser(std::string &result,
                           const std::string &userId,
                           const std::string &projectId,
                           Hanami::ErrorContainer &error);

bool listProjectsOfUser(std::string &result, Hanami::ErrorContainer &error);

bool switchProject(std::string &result,
                   const std::string &projectId,
                   Hanami::ErrorContainer &error);

}  // namespace Hanami

#endif  // HANAMISDK_USER_H
