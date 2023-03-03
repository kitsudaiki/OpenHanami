/**
 * @file        gpu_kernel.cl
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2019 Tobias Anker
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

// const predefined values
#define UNINIT_STATE_64 0xFFFFFFFFFFFFFFFF
#define UNINIT_STATE_32 0xFFFFFFFF
#define UNINIT_STATE_24 0xFFFFFF
#define UNINIT_STATE_16 0xFFFF
#define UNINIT_STATE_8  0xFF

// common information
#define SYNAPSES_PER_SYNAPSESECTION 30
#define NEURONS_PER_NEURONSECTION 63
#define NUMBER_OF_RAND_VALUES 10485760
#define RAND_MAX 2147483647

enum SegmentTypes
{
    UNDEFINED_SEGMENT = 0,
    INPUT_SEGMENT = 1,
    OUTPUT_SEGMENT = 2,
    DYNAMIC_SEGMENT = 3,
};

//==================================================================================================

typedef struct BrickHeader_struct
{
    // common
    uint brickId;
    bool isOutputBrick;
    bool isInputBrick;
    uchar padding1[14];
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

    uchar refractionTime;
    uchar active;
    uchar padding[6];

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
    uchar padding[20];

    // total size: 2048 Byte
} NeuronSection;

//==================================================================================================

typedef struct Synapse_struct
{
    float weight;
    float border;
    ushort targetNeuronId;
    char activeCounter;
    uchar padding[5];
    // total size: 16 Byte
} Synapse;

//==================================================================================================

typedef struct SynapseConnection_struct
{
    uchar active;
    uchar padding[3];

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
    uchar padding[4];
    // total size: 16 Byte
} UpdatePos;

//==================================================================================================

typedef struct UpdatePosSection_struct
{
    UpdatePos positions[NEURONS_PER_NEURONSECTION];
    uint numberOfPositions;
    uchar padding[12];
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
    uchar refractionTime;
    uchar doLearn;
    uchar updateSections;

    uchar padding[213];

    // total size: 256 Byte
} SegmentSettings;

//==================================================================================================

typedef struct NeuronSynapseConnection_struct
{
    uint backwardIds[256];
    // total size: 2048 Byte
} NeuronConnection;


//=========================================================================================================
//=========================================================================================================
//=========================================================================================================

/**
 * @brief initialize a new specific synapse
 */
inline void
createNewSynapse(__global SynapseConnection* connection,
                 __global Synapse* synapse,
                 __global const NeuronSection* targetNeuronSection,
                 __global const SegmentSettings* segmentSettings,
                 const float outH,
                 __global const uint* randomValues)
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

//=========================================================================================================

/**
 * @brief process synapse-section
 */
inline void
synapseProcessingBackward(__global SynapseSection* section,
                          __global SynapseConnection* connection,
                          __global NeuronSection* targetNeuronSection,
                          __global NeuronSection* neuronSections,
                          __global UpdatePosSection* updatePosSections,
                          __global SegmentSettings* segmentSettings,
                          __global const uint* randomValues,
                          __local float* localMem)
{
    __global NeuronSection* sourceNeuronSection = &neuronSections[connection->sourceNeuronSectionId];
    __global Neuron* sourceNeuron = &sourceNeuronSection->neurons[connection->sourceNeuronId];
    const float sourcePotential = sourceNeuron->potential;

    float counter = connection->offset;
    uint pos = 0;

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && sourcePotential > counter)
    {
        __global Synapse* synapse = &section->synapses[pos];

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
        __global Neuron* targetNeuron = &targetNeuronSection->neurons[synapse->targetNeuronId];
        //targetNeuron->input += synapse->weight;
        localMem[synapse->targetNeuronId] += synapse->weight;

        // update active-counter
        const uchar active = (synapse->weight > 0) == (targetNeuron->potential > targetNeuron->border);
        synapse->activeCounter += active * (uchar)(synapse->activeCounter < 126);

        // update loop-counter
        counter += synapse->border;
        pos++;
    }

    __global UpdatePosSection* updateSection = &updatePosSections[connection->sourceNeuronSectionId];
    __global UpdatePos* updatePos = &updateSection->positions[connection->sourceNeuronId];
    updatePos->type = sourcePotential - counter > 0.01f && connection->forwardNextId == UNINIT_STATE_32;
    updatePos->offset = counter + connection->offset;
}

//=========================================================================================================

inline void
prcessNeuronConnection(const uint neuronSectionId,
                       __global NeuronSection* targetNeuronSection,
                       __global NeuronConnection* neuronConnections,
                       __global NeuronSection* neuronSections,
                       __global SynapseConnection* synapseConnections,
                       __global SynapseSection* synapseSections,
                       __global UpdatePosSection* updatePosSections,
                       __global SegmentSettings* segmentSettings,
                       __global const uint* randomValues,
                       __local float* localMem)
{
    for(uint sectionPos = get_local_id(0);
        sectionPos < 256;
        sectionPos += get_local_size(0))
    {
        const uint offset = sectionPos * 64;
        for(uint i = 0; i < 64; i++) {
            localMem[offset + i] = 0.0f;
        }

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
    }
}

//=========================================================================================================

/**
 * @brief process all neurons within a segment
 */
__kernel void
prcessCoreSegment(__global BrickHeader* bricks,
                  __global uint* brickOrder,
                  __global NeuronConnection* neuronConnections,
                  __global NeuronSection* neuronSections,
                  __global SynapseConnection* synapseConnections,
                  __global SynapseSection* synapseSections,
                  __global UpdatePosSection* updatePosSections,
                  __global SegmentSettings* segmentSettings,
                  __global float* inputTransfers,
                  __global float* outputTransfers,
                  __global const uint* randomValues,
                  const ulong numberOfBricks,
                  __local float* localMem)
{
    for(ulong pos = 0; pos < numberOfBricks; pos++)
    {
        __global BrickHeader* brick = &bricks[brickOrder[pos]];
        if(brick->isInputBrick == false
                && brick->isOutputBrick == false)
        {
            for(uint neuronSectionId = brick->neuronSectionPos + get_group_id(0);
                neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
                neuronSectionId += get_num_groups(0))
            {
                __global NeuronSection* neuronSection = &neuronSections[neuronSectionId];

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

                barrier(CLK_LOCAL_MEM_FENCE);

                for(uint neuronId = get_local_id(0);
                    neuronId < neuronSection->numberOfNeurons;
                    neuronId += get_local_size(0))
                {
                    __global Neuron* neuron = &neuronSection->neurons[neuronId];

                    neuron->input = 0.0f;
                    for(uint i = neuronId; i < 64*256; i += 64) {
                        neuron->input += localMem[i];
                    }

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
                    __global UpdatePos* updatePos = &updatePosSections[neuronSectionId].positions[neuronId];
                    updatePos->type = needUpdate;
                    updatePos->offset = 0.0f;
                }
            }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

//=========================================================================================================

__kernel void
prcessOutput(__global BrickHeader* bricks,
             __global NeuronConnection* neuronConnections,
             __global NeuronSection* neuronSections,
             __global SynapseConnection* synapseConnections,
             __global SynapseSection* synapseSections,
             __global UpdatePosSection* updatePosSections,
             __global SegmentSettings* segmentSettings,
             __global float* outputTransfers,
             __global const uint* randomValues,
             __local float* localMem)
{
    __global NeuronSection* neuronSection = &neuronSections[get_group_id(0)];
    __global BrickHeader* brick = &bricks[neuronSection->brickId];
    if(brick->isOutputBrick)
    {
        prcessNeuronConnection(get_group_id(0),
                               neuronSection,
                               neuronConnections,
                               neuronSections,
                               synapseConnections,
                               synapseSections,
                               updatePosSections,
                               segmentSettings,
                               randomValues,
                               localMem);

        barrier(CLK_LOCAL_MEM_FENCE);

        for(uint neuronId = get_local_id(0);
            neuronId < neuronSection->numberOfNeurons;
            neuronId += get_local_size(0))
        {
            __global Neuron* neuron = &neuronSection->neurons[neuronId];
            neuron->input = 0.0f;
            for(uint i = neuronId; i < 64*256; i += 64) {
                neuron->input += localMem[i];
            }

            neuron->potential = segmentSettings->potentialOverflow * neuron->input;
            outputTransfers[neuron->targetBorderId] = neuron->potential;
            neuron->input = 0.0f;
        }
    }
}

//=========================================================================================================

__kernel void
prcessInput(__global BrickHeader* bricks,
            __global NeuronSection* neuronSections,
            __global UpdatePosSection* updatePosSections,
            __global float* inputTransfers)
{
    __global NeuronSection* neuronSection = &neuronSections[get_group_id(0)];
    __global BrickHeader* brick = &bricks[neuronSection->brickId];
    if(brick->isInputBrick
            && get_local_id(0) < neuronSection->numberOfNeurons)
    {
        __global Neuron* neuron = &neuronSection->neurons[get_local_id(0)];
        neuron->potential = inputTransfers[neuron->targetBorderId];
        neuron->active = neuron->potential > 0.0f;

        // handle active-state
        const bool needUpdate = neuron->active != 0 && neuron->targetSectionId == UNINIT_STATE_32;
        __global UpdatePos* updatePos = &updatePosSections[get_group_id(0)].positions[get_local_id(0)];
        updatePos->type = needUpdate;
        updatePos->offset = 0.0f;
    }
}

//=========================================================================================================
//=========================================================================================================
//=========================================================================================================

/**
 * @brief run backpropagation for a single synapse-section
 */
inline uint
backpropagateSection(__global SynapseSection* section,
                     __global SynapseConnection* connection,
                     __global Neuron* sourceNeuron,
                     const float outH,
                     __global const BrickHeader* brick,
                     __global NeuronSection* neuronSections,
                     __global SynapseConnection* synapseConnections,
                     __global SynapseSection* synapseSections)
{
    __global Synapse* synapse = NULL;
    __global Neuron* targetNeuron = NULL;
    __global NeuronSection* targetNeuronSection = &neuronSections[connection->targetNeuronSectionId];
    float learnValue = 0.2f;
    ushort pos = 0;
    float counter = connection->offset;

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && outH > counter)
    {
        // break look, if no more synapses to process
        synapse = &section->synapses[pos];

        // update weight
        learnValue = (float)(126 - synapse->activeCounter) * 0.0002f;
        learnValue += 0.05f;
        targetNeuron = &targetNeuronSection->neurons[synapse->targetNeuronId];
        sourceNeuron->delta += targetNeuron->delta * synapse->weight;
        synapse->weight -= learnValue * targetNeuron->delta;

        counter += synapse->border;
        pos++;
    }

    return connection->forwardNextId;
}

//=========================================================================================================

/**
 * @brief correct weight of synapses within a segment
 */
__kernel void
reweightCoreSegment(__global BrickHeader* bricks,
                    __global uint* brickOrder,
                    __global NeuronSection* neuronSections,
                    __global SynapseConnection* synapseConnections,
                    __global SynapseSection* synapseSections,
                    __global SegmentSettings* segmentSettings,
                    __global float* inputTransfers,
                    __global float* outputTransfers,
                    const ulong numberOfBricks)
{
    // run back-propagation over all internal neurons and synapses
    for(long pos = numberOfBricks - 1; pos >= 0; pos--)
    {
        __global BrickHeader* brick = &bricks[brickOrder[pos]];

        for(uint neuronSectionId = brick->neuronSectionPos + get_group_id(0);
            neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
            neuronSectionId += get_num_groups(0))
        {
            __global NeuronSection* neuronSection = &neuronSections[neuronSectionId];
            for(uint neuronId = get_local_id(0);
                neuronId < neuronSection->numberOfNeurons;
                neuronId += get_local_size(0))
            {
                __global Neuron* sourceNeuron = &neuronSection->neurons[neuronId];
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

        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

//=========================================================================================================

__kernel void
reweightOutput(__global BrickHeader* bricks,
               __global NeuronSection* neuronSections,
               __global float* inputTransfers)
{
    __global NeuronSection* neuronSection = &neuronSections[get_group_id(0)];
    __global BrickHeader* brick = &bricks[neuronSection->brickId];
    if(brick->isOutputBrick)
    {
        if(get_local_id(0) < neuronSection->numberOfNeurons)
        {
            __global Neuron* neuron = &neuronSection->neurons[get_local_id(0)];
            neuron->delta = inputTransfers[neuron->targetBorderId];
            inputTransfers[neuron->targetBorderId] = 0.0f;
        }
    }
}
