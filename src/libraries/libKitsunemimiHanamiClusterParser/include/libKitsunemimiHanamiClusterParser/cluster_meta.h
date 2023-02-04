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

#ifndef KITSUNEMIMI_HANAMI_CLUSTER_PARSER_ITEM_H
#define KITSUNEMIMI_HANAMI_CLUSTER_PARSER_ITEM_H

#include <string>
#include <vector>
#include <map>

#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{
namespace Hanami
{

struct ClusterConnection
{
    std::string sourceBrick = "";
    std::string targetSegment = "";
    std::string targetBrick = "";
};

struct SegmentMetaPtr
{
    std::string name = "";
    std::string type = "";
    std::vector<ClusterConnection> outputs;
};

struct ClusterMeta
{
    uint32_t version = 0;
    std::vector<SegmentMetaPtr> segments;

    SegmentMetaPtr*
    getSegmentMetaPtr(const std::string &name)
    {
        SegmentMetaPtr* tempConnection = nullptr;
        for(uint64_t i = 0; i < segments.size(); i++)
        {
            tempConnection = &segments[i];
            if(tempConnection->name == name) {
                return tempConnection;
            }
        }
        return tempConnection;
    }
};

bool
parseCluster(ClusterMeta* result,
             const std::string &input,
             ErrorContainer &error);

}  // namespace Hanami
}  // namespace Kitsunemimi

#endif // KITSUNEMIMI_HANAMI_CLUSTER_PARSER_ITEM_H
