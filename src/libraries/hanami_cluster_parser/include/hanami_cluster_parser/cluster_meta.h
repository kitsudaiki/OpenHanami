/**
 * @file       cluster_meta.h
 *
 * @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright  Apache License Version 2.0
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

#ifndef HANAMI_SEGMENT_PARSER_ITEM_H
#define HANAMI_SEGMENT_PARSER_ITEM_H

#include <hanami_common/logger.h>
#include <hanami_common/structs.h>

#include <any>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace Hanami
{

struct HexagonMeta {
    Hanami::Position position;
};

struct InputMeta {
    std::string name = "";
    uint32_t targetHexagonId = UNINTI_POINT_32;
    uint64_t numberOfInputs = 0;
};

struct OutputMeta {
    std::string name = "";
    uint32_t targetHexagonId = UNINTI_POINT_32;
    uint64_t numberOfOutputs = 0;
};

struct ClusterMeta {
    uint32_t version = 0;
    float neuronCooldown = 1000000000.f;
    uint32_t refractoryTime = 1;
    uint32_t maxConnectionDistance = 1;
    bool enableReduction = false;

    std::vector<HexagonMeta> hexagons;
    std::vector<InputMeta> inputs;
    std::vector<OutputMeta> outputs;
};

bool parseCluster(ClusterMeta* result, const std::string& input, ErrorContainer& error);

}  // namespace Hanami

#endif  // HANAMI_SEGMENT_PARSER_ITEM_H
