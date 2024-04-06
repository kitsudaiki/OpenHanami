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

    bool restoreConnectionBlocks(Cluster* cluster,
                                 const CheckpointHeader& header,
                                 uint8_t* u8Data,
                                 Hanami::ErrorContainer& error);
};

#endif  // CHECKPOINTIO_H
