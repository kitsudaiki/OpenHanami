/**
 * @file        cluster_init.h
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

#ifndef HANAMI_CLUSTERINIT_H
#define HANAMI_CLUSTERINIT_H

#include <common.h>

#include <libKitsunemimiHanamiClusterParser/segment_meta.h>

class CoreSegment;
class Cluster;

bool reinitPointer(Cluster* cluster, const std::string &uuid);

bool initNewCluster(Cluster* cluster,
                    const Kitsunemimi::Hanami::SegmentMeta &clusterTemplate,
                    const std::string &uuid);

CoreSegment* addDynamicSegment(Cluster* cluster,
                               const Kitsunemimi::Hanami::SegmentMeta &segmentMeta);

#endif // HANAMI_CLUSTERINIT_H
