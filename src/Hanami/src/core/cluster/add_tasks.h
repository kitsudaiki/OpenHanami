/**
 * @file        add_tasks.h
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

#ifndef HANAMI_ADD_TASKS_H
#define HANAMI_ADD_TASKS_H

#include <common.h>
#include <core/cluster/task.h>

class Cluster;

const std::string addImageLearnTask(Cluster &cluster,
                                    const std::string &name,
                                    const std::string &userId,
                                    const std::string &projectId,
                                    float* inputData,
                                    const uint64_t numberOfInputsPerCycle,
                                    const uint64_t numberOfOuputsPerCycle,
                                    const uint64_t numberOfCycle);
const std::string addImageRequestTask(Cluster &cluster,
                                      const std::string &name,
                                      const std::string &userId,
                                      const std::string &projectId,
                                      float* inputData,
                                      const uint64_t numberOfInputsPerCycle,
                                      const uint64_t numberOfOuputsPerCycle,
                                      const uint64_t numberOfCycle);
const std::string addTableLearnTask(Cluster &cluster,
                                    const std::string &name,
                                    const std::string &userId,
                                    const std::string &projectId,
                                    float* inputData,
                                    float* outputData,
                                    const uint64_t numberOfInputs,
                                    const uint64_t numberOfOutputs,
                                    const uint64_t numberOfCycle);
const std::string addTableRequestTask(Cluster &cluster,
                                      const std::string &name,
                                      const std::string &userId,
                                      const std::string &projectId,
                                      float* inputData,
                                      const uint64_t numberOfInputs,
                                      const uint64_t numberOfOutputs,
                                      const uint64_t numberOfCycle);
const std::string addClusterSnapshotSaveTask(Cluster &cluster,
                                             const std::string &snapshotName,
                                             const std::string &userId,
                                             const std::string &projectId);
const std::string addClusterSnapshotRestoreTask(Cluster &cluster,
                                                const std::string &name,
                                                const std::string &snapshotInfo,
                                                const std::string &userId,
                                                const std::string &projectId);

#endif // HANAMI_ADD_TASKS_H
