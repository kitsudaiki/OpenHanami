/**
 * @file        checkpoint_io.h
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

#ifndef CHECKPOINTIO_H
#define CHECKPOINTIO_H

#include <hanami_common/files/binary_file.h>
#include <hanami_common/logger.h>

#include <string>

class Cluster;
struct CheckpointHeader;

class CheckpointIO
{
   public:
    CheckpointIO();

    bool writeClusterToFile(Cluster* cluster,
                            const std::string& filePath,
                            Hanami::ErrorContainer& error);
    bool restoreClusterFromFile(Cluster* cluster,
                                const std::string& fileLocation,
                                Hanami::ErrorContainer& error);

   private:
    bool writeClusterHeaderToFile(Cluster* cluster,
                                  Hanami::BinaryFile& file,
                                  uint64_t& position,
                                  Hanami::ErrorContainer& error);

    bool writeBricksToFile(Cluster* cluster,
                           Hanami::BinaryFile& file,
                           uint64_t& position,
                           Hanami::ErrorContainer& error);

    bool writeNeuronBlocksToFile(Cluster* cluster,
                                 Hanami::BinaryFile& file,
                                 uint64_t& position,
                                 Hanami::ErrorContainer& error);

    bool writeOutputNeuronsToFile(Cluster* cluster,
                                  Hanami::BinaryFile& file,
                                  uint64_t& position,
                                  Hanami::ErrorContainer& error);

    bool writeConnectionBlocksOfBricksToFile(Cluster* cluster,
                                             Hanami::BinaryFile& file,
                                             uint64_t& position,
                                             Hanami::ErrorContainer& error);

    bool writeConnectionBlockToFile(Cluster* cluster,
                                    Hanami::BinaryFile& file,
                                    uint64_t& position,
                                    const uint64_t brickId,
                                    const uint64_t blockid,
                                    Hanami::ErrorContainer& error);

    bool writeSynapseBlockToFile(Cluster* cluster,
                                 Hanami::BinaryFile& file,
                                 uint64_t& position,
                                 const uint64_t targetSynapseBlockPos,
                                 Hanami::ErrorContainer& error);

    bool restoreClusterHeader(Cluster* cluster,
                              const CheckpointHeader& header,
                              uint8_t* u8Data,
                              Hanami::ErrorContainer& error);

    bool restoreBricks(Cluster* cluster,
                       const CheckpointHeader& header,
                       uint8_t* u8Data,
                       Hanami::ErrorContainer& error);

    bool restoreNeuronBlocks(Cluster* cluster,
                             const CheckpointHeader& header,
                             uint8_t* u8Data,
                             Hanami::ErrorContainer& error);

    bool restoreOutputNeurons(Cluster* cluster,
                              const CheckpointHeader& header,
                              uint8_t* u8Data,
                              Hanami::ErrorContainer& error);

    bool restoreConnectionBlocks(Cluster* cluster,
                                 const CheckpointHeader& header,
                                 uint8_t* u8Data,
                                 Hanami::ErrorContainer& error);

    uint64_t getDataSize(Cluster* cluster) const;
};

#endif  // CHECKPOINTIO_H
