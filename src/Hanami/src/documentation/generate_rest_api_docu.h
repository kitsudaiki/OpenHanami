/**
 * @file        generate_rest_api_docu.h
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

#ifndef HANAMI_GENERATERESTAPIDOCU_H
#define HANAMI_GENERATERESTAPIDOCU_H

#include <common.h>

#include <map>
#include <string>

struct FieldDef;

void createOpenApiDocumentation(std::string &docu);

void generateEndpointDocu_openapi(json &result);
void createBodyParams_openapi(json &schema,
                              const std::map<std::string, FieldDef> *defMap,
                              const bool isRequest);
void createQueryParams_openapi(json &parameters, const std::map<std::string, FieldDef> *defMap);
void addTokenRequirement(json &parameters);

#endif  // HANAMI_GENERATERESTAPIDOCU_H
