/**
 * @file        project.h
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

#ifndef KITSUNEMIMI_HANAMISDK_PROJECT_H
#define KITSUNEMIMI_HANAMISDK_PROJECT_H

#include <libKitsunemimiCommon/logger.h>

namespace HanamiAI
{

bool createProject(std::string &result,
                   const std::string &projectId,
                   const std::string &projectName,
                   Kitsunemimi::ErrorContainer &error);

bool getProject(std::string &result,
                const std::string &projectId,
                Kitsunemimi::ErrorContainer &error);

bool listProject(std::string &result,
                 Kitsunemimi::ErrorContainer &error);

bool deleteProject(std::string &result,
                   const std::string &projectId,
                   Kitsunemimi::ErrorContainer &error);

} // namespace HanamiAI

#endif // KITSUNEMIMI_HANAMISDK_PROJECT_H
