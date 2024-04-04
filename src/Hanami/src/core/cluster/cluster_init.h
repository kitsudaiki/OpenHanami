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
#include <core/processing/objects.h>
#include <hanami_cluster_parser/cluster_meta.h>

class CoreSegment;
class Cluster;

using Hanami::ClusterMeta;

bool reinitPointer(Cluster* cluster, const uint64_t numberOfBytes);

bool initNewCluster(Cluster* cluster,
                    const Hanami::ClusterMeta& clusterMeta,
                    const std::string& uuid);

void initSegmentPointer(Cluster* cluster);
bool connectBorderBuffer(Cluster* cluster);
// void initOpencl(Hanami::GpuData &data);
void initSettings(Cluster* cluster, const Hanami::ClusterMeta& clusterMeta);

void addBricksToCluster(Cluster* cluster, const Hanami::ClusterMeta& clusterMeta);
bool initTargetBrickList(Cluster* cluster);

Brick createNewBrick(const Hanami::BrickMeta& brickMeta, const uint32_t id);
void connectBrick(Cluster* cluster, Brick& sourceBrick, const uint8_t side);
void connectAllBricks(Cluster* cluster);
bool initializeNeurons(Cluster* cluster, const Hanami::ClusterMeta& clusterMeta);
uint32_t goToNextInitBrick(Cluster* cluster, Brick& currentBrick, uint32_t* maxPathLength);

#endif  // HANAMI_CLUSTERINIT_H
