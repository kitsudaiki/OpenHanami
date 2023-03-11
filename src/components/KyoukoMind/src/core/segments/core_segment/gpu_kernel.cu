#include <iostream>
#include <chrono>

#define UUID_STR_LEN	37

// const predefined values
#define UNINIT_STATE_64 0xFFFFFFFFFFFFFFFF
#define UNINIT_STATE_32 0xFFFFFFFF
#define UNINIT_STATE_24 0xFFFFFF
#define UNINIT_STATE_16 0xFFFF
#define UNINIT_STATE_8  0xFF
#define UNINTI_POINT_32 0x0FFFFFFF

// common information
#define SYNAPSES_PER_SYNAPSESECTION 30
#define NEURONS_PER_NEURONSECTION 63
#define NEURON_CONNECTIONS 512
#define NUMBER_OF_RAND_VALUES 10485760
#define RAND_MAX 2147483647

enum SegmentTypes
{
    UNDEFINED_SEGMENT = 0,
    INPUT_SEGMENT = 1,
    OUTPUT_SEGMENT = 2,
    CORE_SEGMENT = 3,
};

enum ObjectTypes
{
    CLUSTER_OBJECT = 0,
    SEGMENT_OBJECT = 1,
};

struct SegmentHeaderEntry
{
    uint64_t bytePos = 0;
    uint64_t count = 0;

    // total size: 16 Byte
};

struct kuuid
{
    char uuid[UUID_STR_LEN];
    uint8_t padding[3];

    // total size: 40 Bytes
};

struct Position
{
    uint32_t x = UNINTI_POINT_32;
    uint32_t y = UNINTI_POINT_32;
    uint32_t z = UNINTI_POINT_32;
    uint32_t w = UNINTI_POINT_32;
};

struct SegmentHeader
{
    uint8_t objectType = SEGMENT_OBJECT;
    uint8_t version = 1;
    uint8_t segmentType = UNDEFINED_SEGMENT;
    uint8_t padding;
    uint32_t segmentID = UNINIT_STATE_32;
    uint64_t staticDataSize = 0;
    Position position;

    kuuid parentClusterId;

    // synapse-segment
    SegmentHeaderEntry name;
    SegmentHeaderEntry settings;
    SegmentHeaderEntry slotList;
    SegmentHeaderEntry inputTransfers;
    SegmentHeaderEntry outputTransfers;

    SegmentHeaderEntry bricks;
    SegmentHeaderEntry brickOrder;
    SegmentHeaderEntry neuronSections;
    SegmentHeaderEntry inputs;
    SegmentHeaderEntry outputs;
    SegmentHeaderEntry updatePosSections;

    SegmentHeaderEntry synapseSections;

    uint8_t padding2[246];

    // total size: 512 Byte
};

//==================================================================================================

typedef struct BrickHeader_struct
{
    // common
    uint brickId;
    bool isOutputBrick;
    bool isInputBrick;
    uint8_t padding1[14];
    uint neuronSectionPos;
    uint numberOfNeurons;
    uint numberOfNeuronSections;

    // total size: 32 Bytes
} BrickHeader;

//==================================================================================================

typedef struct Neuron_struct
{
    float input;
    float border;
    float potential;
    float delta;

    uint8_t refractionTime;
    uint8_t active;
    uint8_t padding[6];

    uint targetBorderId;
    uint targetSectionId;

    // total size: 32 Byte
} Neuron;

//==================================================================================================

typedef struct NeuronSection_struct
{
    Neuron neurons[NEURONS_PER_NEURONSECTION];
    uint numberOfNeurons;
    uint brickId;
    uint backwardNextId;
    uint8_t padding[20];

    // total size: 2048 Byte
} NeuronSection;

//==================================================================================================

typedef struct Synapse_struct
{
    float weight;
    float border;
    ushort targetNeuronId;
    char activeCounter;
    uint8_t padding[5];
    // total size: 16 Byte
} Synapse;

//==================================================================================================

typedef struct SynapseConnection_struct
{
    uint8_t active;
    uint8_t padding[3];

    float offset;
    uint randomPos;

    uint forwardNextId;
    uint backwardNextId;

    uint targetNeuronSectionId;
    uint sourceNeuronSectionId;
    uint sourceNeuronId;

    // total size: 32 Byte
} SynapseConnection;

//==================================================================================================

typedef struct SynapseSection_struct
{
    SynapseConnection connection;

    Synapse synapses[SYNAPSES_PER_SYNAPSESECTION];
    // total size: 512 Byte
} SynapseSection;

//==================================================================================================

typedef struct UpdatePos_struct
{
    uint type;
    uint randomPos;
    float offset;
    uint8_t padding[4];
    // total size: 16 Byte
} UpdatePos;

//==================================================================================================

typedef struct UpdatePosSection_struct
{
    UpdatePos positions[NEURONS_PER_NEURONSECTION];
    uint numberOfPositions;
    uint8_t padding[12];
    // total size: 1024 Byte
} UpdatePosSection;

//==================================================================================================

typedef struct SegmentSettings
{
    ulong maxSynapseSections;
    float synapseDeleteBorder;
    float neuronCooldown;
    float memorizing;
    float gliaValue;
    float signNeg;
    float potentialOverflow;
    float synapseSegmentation;
    float backpropagationBorder;
    uint8_t refractionTime;
    uint8_t doLearn;
    uint8_t updateSections;

    uint8_t padding[213];

    // total size: 256 Byte
} SegmentSettings;

//==================================================================================================

typedef struct NeuronSynapseConnection_struct
{
    uint backwardIds[NEURON_CONNECTIONS];
    // total size: 2048 Byte
} NeuronConnection;


//==================================================================================================
//==================================================================================================
//==================================================================================================

__device__ __forceinline__ int
getBlockId()
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    return index / blockDim.x;
}

/**
 * @brief initialize a new specific synapse
 */
__device__ __forceinline__ void
createNewSynapse(SynapseConnection* connection,
                 Synapse* synapse,
                 const NeuronSection* targetNeuronSection,
                 const SegmentSettings* segmentSettings,
                 const float outH,
                 const uint* randomValues)
{
    const float maxWeight = outH / (float)(segmentSettings->synapseSegmentation);

    // set activation-border
    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->border = maxWeight * ((float)(randomValues[connection->randomPos]) / (float)(RAND_MAX));

    // set target neuron
    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->targetNeuronId = (ushort)(randomValues[connection->randomPos]
                              % targetNeuronSection->numberOfNeurons);


    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->weight = ((float)(randomValues[connection->randomPos]) / (float)(RAND_MAX)) / 10.0f;

    // update weight with sign
    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    const uint signRand = randomValues[connection->randomPos] % 1000;
    synapse->weight *= (float)(1.0f - (1000.0f * segmentSettings->signNeg > signRand) * 2);

    synapse->activeCounter = 1;
}

//==================================================================================================

/**
 * @brief process synapse-section
 */
__device__ __forceinline__ void
synapseProcessingBackward(SynapseSection* section,
                          SynapseConnection* connection,
                          NeuronSection* targetNeuronSection,
                          NeuronSection* neuronSections,
                          UpdatePosSection* updatePosSections,
                          SegmentSettings* segmentSettings,
                          const uint* randomValues,
                          float* localMem)
{
    NeuronSection* sourceNeuronSection = &neuronSections[connection->sourceNeuronSectionId];
    Neuron* sourceNeuron = &sourceNeuronSection->neurons[connection->sourceNeuronId];
    const float sourcePotential = sourceNeuron->potential;

    float counter = connection->offset;
    uint pos = 0;

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && sourcePotential > counter)
    {
        Synapse* synapse = &section->synapses[pos];

        // create new synapse if necesarry and learning is active
        if(synapse->targetNeuronId == UNINIT_STATE_16)
        {
            createNewSynapse(connection,
                             synapse,
                             targetNeuronSection,
                             segmentSettings,
                             sourcePotential,
                             randomValues);
        }

        // update target-neuron
        Neuron* targetNeuron = &targetNeuronSection->neurons[synapse->targetNeuronId];
        //targetNeuron->input += synapse->weight;
        localMem[synapse->targetNeuronId] += synapse->weight;

        // update active-counter
        const uint8_t active = (synapse->weight > 0) == (targetNeuron->potential > targetNeuron->border);
        synapse->activeCounter += active * (uint8_t)(synapse->activeCounter < 126);

        // update loop-counter
        counter += synapse->border;
        pos++;
    }

    UpdatePosSection* updateSection = &updatePosSections[connection->sourceNeuronSectionId];
    UpdatePos* updatePos = &updateSection->positions[connection->sourceNeuronId];
    updatePos->type = sourcePotential - counter > 0.01f && connection->forwardNextId == UNINIT_STATE_32;
    updatePos->offset = counter + connection->offset;
}

//==================================================================================================

__device__ __forceinline__ void
prcessNeuronConnection(const uint neuronSectionId,
                       NeuronSection* targetNeuronSection,
                       NeuronConnection* neuronConnections,
                       NeuronSection* neuronSections,
                       SynapseConnection* synapseConnections,
                       SynapseSection* synapseSections,
                       UpdatePosSection* updatePosSections,
                       SegmentSettings* segmentSettings,
                       const uint* randomValues,
                       float* localMem)
{
    // reset weight of neurons
    for(uint neuronId = threadIdx.x;
        neuronId < targetNeuronSection->numberOfNeurons;
        neuronId += blockDim.x)
    {
        targetNeuronSection->neurons[neuronId].input = 0.0f;
    }

    for(uint sectionPos = threadIdx.x;
        sectionPos < NEURON_CONNECTIONS;
        sectionPos += blockDim.x)
    {
        // process synapse-sections
        const uint offset = threadIdx.x * NEURONS_PER_NEURONSECTION;
        const uint sectionId = neuronConnections[neuronSectionId].backwardIds[sectionPos];
        if(sectionId != UNINIT_STATE_32)
        {
            synapseProcessingBackward(&synapseSections[sectionId],
                                      &synapseConnections[sectionId],
                                      targetNeuronSection,
                                      neuronSections,
                                      updatePosSections,
                                      segmentSettings,
                                      randomValues,
                                      &localMem[offset]);
        }

        __syncthreads();

        // apply values of the local-memory to the neurons
        for(uint neuronId = threadIdx.x;
            neuronId < targetNeuronSection->numberOfNeurons;
            neuronId += blockDim.x)
        {
            Neuron* neuron = &targetNeuronSection->neurons[neuronId];
            for(uint i = neuronId;
                i < NEURONS_PER_NEURONSECTION * blockDim.x;
                i += NEURONS_PER_NEURONSECTION)
            {
                neuron->input += localMem[i];
                localMem[i] = 0.0f;
            }
        }
    }

    __syncthreads();
}

//==================================================================================================

__device__ __forceinline__ void
resetLocalMemory(float* localMem, const int localSize)
{
    // reset local memory
    for(uint i = threadIdx.x;
        i < localSize;
        i += blockDim.x)
    {
        localMem[i] = 0.0f;
    }
}

//==================================================================================================

/**
 * @brief process all neurons within a segment
 */
__global__ void
prcessCoreSegmentKernel(BrickHeader* bricks,
                        NeuronConnection* neuronConnections,
                        NeuronSection* neuronSections,
                        SynapseConnection* synapseConnections,
                        SynapseSection* synapseSections,
                        UpdatePosSection* updatePosSections,
                        SegmentSettings* segmentSettings,
                        float* inputTransfers,
                        float* outputTransfers,
                        const uint* randomValues,
                        const ulong brickId)
{
    __shared__ float localMem[4096];
    resetLocalMemory(localMem, blockDim.x * NEURONS_PER_NEURONSECTION);

    BrickHeader* brick = &bricks[brickId];
    if(brick->isInputBrick == false
            && brick->isOutputBrick == false)
    {
        for(uint neuronSectionId = brick->neuronSectionPos + getBlockId();
            neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
            neuronSectionId += gridDim.x)
        {
            NeuronSection* neuronSection = &neuronSections[neuronSectionId];

            prcessNeuronConnection(neuronSectionId,
                                   neuronSection,
                                   neuronConnections,
                                   neuronSections,
                                   synapseConnections,
                                   synapseSections,
                                   updatePosSections,
                                   segmentSettings,
                                   randomValues,
                                   localMem);

            for(uint neuronId = threadIdx.x;
                neuronId < neuronSection->numberOfNeurons;
                neuronId += blockDim.x)
            {
                Neuron* neuron = &neuronSection->neurons[neuronId];

                neuron->potential /= segmentSettings->neuronCooldown;
                neuron->refractionTime = neuron->refractionTime >> 1;

                if(neuron->refractionTime == 0)
                {
                    neuron->potential = segmentSettings->potentialOverflow * neuron->input;
                    neuron->refractionTime = segmentSettings->refractionTime;
                }

                // update neuron
                neuron->potential -= neuron->border;
                neuron->active = neuron->potential > 0.0f;
                neuron->input = 0.0f;
                neuron->potential = log2(neuron->potential + 1.0f);

                // handle active-state
                const bool needUpdate = neuron->active != 0 && neuron->targetSectionId == UNINIT_STATE_32;
                UpdatePos* updatePos = &updatePosSections[neuronSectionId].positions[neuronId];
                updatePos->type = needUpdate;
                updatePos->offset = 0.0f;
            }
        }
    }
}

//==================================================================================================

__global__ void
prcessOutputKernel(BrickHeader* bricks,
                   NeuronConnection* neuronConnections,
                   NeuronSection* neuronSections,
                   SynapseConnection* synapseConnections,
                   SynapseSection* synapseSections,
                   UpdatePosSection* updatePosSections,
                   SegmentSettings* segmentSettings,
                   float* outputTransfers,
                   const uint* randomValues)
{
    __shared__ float localMem[4096];
    resetLocalMemory(localMem, blockDim.x * NEURONS_PER_NEURONSECTION);

    NeuronSection* neuronSection = &neuronSections[getBlockId()];
    BrickHeader* brick = &bricks[neuronSection->brickId];
    if(brick->isOutputBrick)
    {
        prcessNeuronConnection(getBlockId(),
                               neuronSection,
                               neuronConnections,
                               neuronSections,
                               synapseConnections,
                               synapseSections,
                               updatePosSections,
                               segmentSettings,
                               randomValues,
                               localMem);

        for(uint neuronId = threadIdx.x;
            neuronId < neuronSection->numberOfNeurons;
            neuronId += blockDim.x)
        {
            Neuron* neuron = &neuronSection->neurons[neuronId];

            neuron->potential = segmentSettings->potentialOverflow * neuron->input;
            outputTransfers[neuron->targetBorderId] = neuron->potential;
            neuron->input = 0.0f;
        }
    }
}

//==================================================================================================

__global__ void
prcessInputKernel(BrickHeader* bricks,
                  NeuronSection* neuronSections,
                  UpdatePosSection* updatePosSections,
                  float* inputTransfers)
{
    NeuronSection* neuronSection = &neuronSections[getBlockId()];
    BrickHeader* brick = &bricks[neuronSection->brickId];
    const int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    if(brick->isInputBrick
            && threadIdx.x < neuronSection->numberOfNeurons)
    {
        Neuron* neuron = &neuronSection->neurons[threadIdx.x];
        neuron->potential = inputTransfers[neuron->targetBorderId];
        neuron->active = neuron->potential > 0.0f;

        // handle active-state
        const bool needUpdate = neuron->active != 0 && neuron->targetSectionId == UNINIT_STATE_32;
        UpdatePos* updatePos = &updatePosSections[getBlockId()].positions[threadIdx.x];
        updatePos->type = needUpdate;
        updatePos->offset = 0.0f;
    }
}

//==================================================================================================
//==================================================================================================
//==================================================================================================

/**
 * @brief run backpropagation for a single synapse-section
 */
__device__ __forceinline__ uint
backpropagateSection(SynapseSection* section,
                     SynapseConnection* connection,
                     Neuron* sourceNeuron,
                     const float outH,
                     const BrickHeader* brick,
                     NeuronSection* neuronSections,
                     SynapseConnection* synapseConnections,
                     SynapseSection* synapseSections)
{
    NeuronSection* targetNeuronSection = &neuronSections[connection->targetNeuronSectionId];
    float learnValue = 0.2f;
    float counter = connection->offset;

    // iterate over all synapses in the section
    for(uint32_t pos = 0; pos < SYNAPSES_PER_SYNAPSESECTION; pos++)
    {
        // break look, if no more synapses to process
        Synapse* synapse = &section->synapses[pos];

        if(outH > counter)
        {
            // update weight
            learnValue = (float)(126 - synapse->activeCounter) * 0.0002f;
            learnValue += 0.05f;
            Neuron* targetNeuron = &targetNeuronSection->neurons[synapse->targetNeuronId];
            sourceNeuron->delta += targetNeuron->delta * synapse->weight;

            synapse->weight -= learnValue * targetNeuron->delta;
        }

        counter += synapse->border;
    }

    return connection->forwardNextId;
}

//==================================================================================================

/**
 * @brief correct weight of synapses within a segment
 */
__global__ void
reweightCoreSegmentKernel(BrickHeader* bricks,
                          NeuronSection* neuronSections,
                          SynapseConnection* synapseConnections,
                          SynapseSection* synapseSections,
                          SegmentSettings* segmentSettings,
                          float* inputTransfers,
                          float* outputTransfers,
                          const ulong brickId)
{
    BrickHeader* brick = &bricks[brickId];

    for(uint neuronSectionId = brick->neuronSectionPos + getBlockId();
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
        neuronSectionId += gridDim.x)
    {
        NeuronSection* neuronSection = &neuronSections[neuronSectionId];
        for(uint neuronId = threadIdx.x;
            neuronId < neuronSection->numberOfNeurons;
            neuronId += blockDim.x)
        {
            Neuron* sourceNeuron = &neuronSection->neurons[neuronId];
            if(sourceNeuron->targetSectionId != UNINIT_STATE_32)
            {
                sourceNeuron->delta = 0.0f;
                if(sourceNeuron->active)
                {
                    uint nextId = sourceNeuron->targetSectionId;
                    while(nextId != UNINIT_STATE_32)
                    {
                        nextId = backpropagateSection(&synapseSections[nextId],
                                                      &synapseConnections[nextId],
                                                      sourceNeuron,
                                                      sourceNeuron->potential,
                                                      brick,
                                                      neuronSections,
                                                      synapseConnections,
                                                      synapseSections);
                    }

                    sourceNeuron->delta *= 1.4427f * pow(0.5f, sourceNeuron->potential);
                }

                if(brick->isInputBrick) {
                    outputTransfers[sourceNeuron->targetBorderId] = sourceNeuron->delta;
                }
            }
        }
    }
}

//==================================================================================================

__global__ void
reweightOutputKernel(BrickHeader* bricks,
                     NeuronSection* neuronSections,
                     float* inputTransfers)
{
    NeuronSection* neuronSection = &neuronSections[getBlockId()];
    BrickHeader* brick = &bricks[neuronSection->brickId];
    if(brick->isOutputBrick
            && threadIdx.x < neuronSection->numberOfNeurons)
    {
        Neuron* neuron = &neuronSection->neurons[threadIdx.x];
        neuron->delta = inputTransfers[neuron->targetBorderId];
        inputTransfers[neuron->targetBorderId] = 0.0f;
    }
}

struct PointerHandler
{
    BrickHeader* bricks = nullptr;
    uint32_t* brickOrder = nullptr;
    NeuronSection* neuronSections = nullptr;
    SynapseSection* synapseSections = nullptr;
    SegmentSettings* segmentSettings = nullptr;
    float* inputTransfers = nullptr;
    float* outputTransfers = nullptr;
    UpdatePosSection* updatePosSections = nullptr;
    uint32_t* randomValues = nullptr;
    NeuronConnection* neuronConnections = nullptr;
    SynapseConnection* synapseConnections = nullptr;
};

extern "C"
void
copyToDevice_CUDA(PointerHandler* gpuPointer,
                  SegmentHeader* segmentHeader,
                  SegmentSettings* segmentSettings,
                  BrickHeader* brickHeaders,
                  uint32_t* brickOrder,
                  NeuronSection* neuronSections,
                  SynapseSection* synapseSections,
                  UpdatePosSection* updatePosSections,
                  SynapseConnection* synapseConnections,
                  NeuronConnection* neuronConnections,
                  float* inputTransfers,
                  float* outputTransfers,
                  uint32_t* randomValues)
{
    cudaMalloc(&gpuPointer->bricks,             segmentHeader->bricks.count             * sizeof(BrickHeader));
    cudaMalloc(&gpuPointer->brickOrder,         segmentHeader->brickOrder.count         * sizeof(uint32_t));
    cudaMalloc(&gpuPointer->neuronSections,     segmentHeader->neuronSections.count     * sizeof(NeuronSection));
    cudaMalloc(&gpuPointer->synapseSections,    segmentHeader->synapseSections.count    * sizeof(SynapseSection));
    cudaMalloc(&gpuPointer->segmentSettings,    1                                       * sizeof(SegmentSettings));
    cudaMalloc(&gpuPointer->inputTransfers,     segmentHeader->inputTransfers.count     * sizeof(float));
    cudaMalloc(&gpuPointer->outputTransfers,    segmentHeader->outputTransfers.count    * sizeof(float));
    cudaMalloc(&gpuPointer->updatePosSections,  segmentHeader->updatePosSections.count  * sizeof(UpdatePosSection));
    cudaMalloc(&gpuPointer->randomValues,       NUMBER_OF_RAND_VALUES                   * sizeof(uint32_t));
    cudaMalloc(&gpuPointer->neuronConnections,  segmentHeader->neuronSections.count     * sizeof(NeuronConnection));
    cudaMalloc(&gpuPointer->synapseConnections, segmentHeader->synapseSections.count    * sizeof(SynapseConnection));

    cudaMemcpy(gpuPointer->bricks,             brickHeaders,       segmentHeader->bricks.count            * sizeof(BrickHeader),       cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->brickOrder,         brickOrder,         segmentHeader->brickOrder.count        * sizeof(uint32_t),          cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->neuronSections,     neuronSections,     segmentHeader->neuronSections.count    * sizeof(NeuronSection),     cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->synapseSections,    synapseSections,    segmentHeader->synapseSections.count   * sizeof(SynapseSection),    cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->segmentSettings,    segmentSettings,    1                                      * sizeof(SegmentSettings),   cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->inputTransfers,     inputTransfers,     segmentHeader->inputTransfers.count    * sizeof(float),             cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->outputTransfers,    outputTransfers,    segmentHeader->outputTransfers.count   * sizeof(float),             cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->updatePosSections,  updatePosSections,  segmentHeader->updatePosSections.count * sizeof(UpdatePosSection),  cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->randomValues,       randomValues,       NUMBER_OF_RAND_VALUES                  * sizeof(uint32_t),          cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->neuronConnections,  neuronConnections,  segmentHeader->neuronSections.count    * sizeof(NeuronConnection),  cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPointer->synapseConnections, synapseConnections, segmentHeader->synapseSections.count   * sizeof(SynapseConnection), cudaMemcpyHostToDevice);
}


extern "C"
void
processing_CUDA(PointerHandler* gpuPointer,
                SegmentHeader* segmentHeader,
                uint32_t* brickOrder,
                BrickHeader* bricks,
                float* inputTransfers,
                float* outputTransfers,
                const uint32_t numberOfNeuronSections)
{
    cudaMemcpy(gpuPointer->inputTransfers,
               inputTransfers,
               segmentHeader->inputTransfers.count * sizeof(float),
               cudaMemcpyHostToDevice);

    prcessInputKernel<<<numberOfNeuronSections, NEURONS_PER_NEURONSECTION>>>(
        gpuPointer->bricks,
        gpuPointer->neuronSections,
        gpuPointer->updatePosSections,
        gpuPointer->inputTransfers);

    const uint32_t numberOfBricks = segmentHeader->bricks.count;
    for(uint32_t pos = 0; pos < numberOfBricks; pos++)
    {
        const uint32_t brickId = brickOrder[pos];
        BrickHeader* brick = &bricks[brickId];
        if(brick->isInputBrick == false
                && brick->isOutputBrick == false)
        {
            prcessCoreSegmentKernel<<<10, 64>>>(
                gpuPointer->bricks,
                gpuPointer->neuronConnections,
                gpuPointer->neuronSections,
                gpuPointer->synapseConnections,
                gpuPointer->synapseSections,
                gpuPointer->updatePosSections,
                gpuPointer->segmentSettings,
                gpuPointer->inputTransfers,
                gpuPointer->outputTransfers,
                gpuPointer->randomValues,
                brickId);
        }
    }

    prcessOutputKernel<<<numberOfNeuronSections, 64>>>(
        gpuPointer->bricks,
        gpuPointer->neuronConnections,
        gpuPointer->neuronSections,
        gpuPointer->synapseConnections,
        gpuPointer->synapseSections,
        gpuPointer->updatePosSections,
        gpuPointer->segmentSettings,
        gpuPointer->outputTransfers,
        gpuPointer->randomValues);

    cudaDeviceSynchronize();
    cudaMemcpy(outputTransfers,
               gpuPointer->outputTransfers,
               segmentHeader->outputTransfers.count * sizeof(float),
               cudaMemcpyDeviceToHost);
}

extern "C"
void
backpropagation_CUDA(PointerHandler* gpuPointer,
                     SegmentHeader* segmentHeader,
                     uint32_t* brickOrder,
                     BrickHeader* bricks,
                     float* inputTransfers,
                     float* outputTransfers,
                     UpdatePosSection* updatePosSections,
                     const uint32_t numberOfNeuronSections)
{
    cudaMemcpy(gpuPointer->inputTransfers,
               inputTransfers,
               segmentHeader->inputTransfers.count * sizeof(float),
               cudaMemcpyHostToDevice);

    reweightOutputKernel<<<numberOfNeuronSections, NEURONS_PER_NEURONSECTION>>> (
        gpuPointer->bricks,
        gpuPointer->neuronSections,
        gpuPointer->inputTransfers);

    const uint32_t numberOfBricks = segmentHeader->bricks.count;
    for(int32_t pos = numberOfBricks - 1; pos >= 0; pos--)
    {
        const uint32_t brickId = brickOrder[pos];
        reweightCoreSegmentKernel<<<numberOfNeuronSections, 64>>>(
            gpuPointer->bricks,
            gpuPointer->neuronSections,
            gpuPointer->synapseConnections,
            gpuPointer->synapseSections,
            gpuPointer->segmentSettings,
            gpuPointer->inputTransfers,
            gpuPointer->outputTransfers,
            brickId);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(outputTransfers,
               gpuPointer->outputTransfers,
               segmentHeader->outputTransfers.count * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(updatePosSections,
               gpuPointer->updatePosSections,
               segmentHeader->updatePosSections.count * sizeof(UpdatePosSection),
               cudaMemcpyDeviceToHost);
}

extern "C"
void
update_CUDA(PointerHandler* gpuPointer,
            SegmentHeader* segmentHeader,
            UpdatePosSection* updatePosSections,
            NeuronSection* neuronSections,
            SynapseConnection* synapseConnections,
            NeuronConnection* neuronConnections)
{
    cudaMemcpy(gpuPointer->updatePosSections,
               updatePosSections,
               segmentHeader->updatePosSections.count * sizeof(UpdatePosSection),
               cudaMemcpyHostToDevice);

    cudaMemcpy(gpuPointer->neuronSections,
               neuronSections,
               segmentHeader->neuronSections.count * sizeof(NeuronSection),
               cudaMemcpyHostToDevice);

    cudaMemcpy(gpuPointer->synapseConnections,
               synapseConnections,
               segmentHeader->synapseSections.count * sizeof(SynapseConnection),
               cudaMemcpyHostToDevice);

    cudaMemcpy(gpuPointer->neuronConnections,
               neuronConnections,
               segmentHeader->neuronSections.count * sizeof(NeuronConnection),
               cudaMemcpyHostToDevice);
}
