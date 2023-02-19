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

typedef struct kuuid_struct
{
    char uuid[37];
    uchar padding[3];
    // total size: 40 Bytes
} kuuid;


typedef struct Position_struct
{
    uint x;
    uint y;
    uint z;
    uint w;
} Position;

enum SegmentTypes
{
    UNDEFINED_SEGMENT = 0,
    INPUT_SEGMENT = 1,
    OUTPUT_SEGMENT = 2,
    DYNAMIC_SEGMENT = 3,
};

typedef struct SegmentHeaderEntry
{
    ulong bytePos;
    ulong count;

    // total size: 16 Byte
} SegmentHeaderEntry;

typedef struct SegmentHeader_struct
{
    uchar objectType;
    uchar version;
    uchar segmentType;
    uchar padding;
    uint segmentID;
    ulong staticDataSize;
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

    uchar padding2[246];

    // total size: 512 Byte
} SegmentHeader;

//==================================================================================================

typedef struct Brick_struct
{
    // common
    uint brickId;
    bool isOutputBrick;
    bool isInputBrick;
    uchar padding1[14];
    uint neuronSectionPos;

    Position brickPos;
    uint neighbors[12];

    uint possibleTargetNeuronBrickIds[1000];
    uint numberOfNeurons;
    uint numberOfNeuronSections;

    // total size: 4096 Bytes
} Brick;

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

typedef struct SectionConnection_struct
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
} SectionConnection;

//==================================================================================================

typedef struct SynapseSection_struct
{
    SectionConnection connection;

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

/**
 * @brief initialize a new specific synapse
 */
inline void
createNewSynapse(__global SectionConnection* connection,
                 __global Synapse* synapse,
                 __global const NeuronSection* targetNeuronSection,
                 __global const SegmentSettings* segmentSettings,
                 const float outH,
                 __global const uint* randomValues)
{
    const float randMax = (float)(RAND_MAX);
    const float maxWeight = outH / (float)(segmentSettings->synapseSegmentation);
    uint signRand = 0;
    const float sigNeg = segmentSettings->signNeg;

    // set activation-border
    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->border = maxWeight * ((float)(randomValues[connection->randomPos]) / randMax);

    // set target neuron
    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->targetNeuronId = (ushort)(randomValues[connection->randomPos]
                              % targetNeuronSection->numberOfNeurons);


    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->weight = ((float)(randomValues[connection->randomPos]) / randMax) / 10.0f;

    // update weight with sign
    connection->randomPos = (connection->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    signRand = randomValues[connection->randomPos] % 1000;
    synapse->weight *= (float)(1.0f - (1000.0f * sigNeg > signRand) * 2);

    synapse->activeCounter = 1;
}

/**
 * @brief process synapse-section
 */
inline void
synapseProcessingBackbard(__global SynapseSection* section,
                          __global SectionConnection* connection,
                          __global NeuronSection* targetNeuronSection,
                          __global NeuronSection* neuronSections,
                          __global SectionConnection* sectionConnections,
                          __global SynapseSection* synapseSections,
                          __global UpdatePosSection* updatePosSections,
                          __global SegmentSettings* segmentSettings,
                          __global const uint* randomValues)
{
    uint pos = 0;
    __global Synapse* synapse = NULL;
    __global Neuron* targetNeuron = NULL;
    uchar active = 0;
    float counter = connection->offset;
    __global Neuron* sourceNeuron = &neuronSections[connection->sourceNeuronSectionId].neurons[connection->sourceNeuronId];
    const float outH = sourceNeuron->potential;

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && outH > counter)
    {
        synapse = &section->synapses[pos];

        // create new synapse if necesarry and learning is active
        if(synapse->targetNeuronId == UNINIT_STATE_16)
        {
            createNewSynapse(connection,
                             synapse,
                             targetNeuronSection,
                             segmentSettings,
                             outH,
                             randomValues);
        }

        // update target-neuron
        targetNeuron = &targetNeuronSection->neurons[synapse->targetNeuronId];
        targetNeuron->input += synapse->weight;

        // update active-counter
        active = (synapse->weight > 0) == (targetNeuron->potential > targetNeuron->border);
        synapse->activeCounter += active * (uchar)(synapse->activeCounter < 126);

        // update loop-counter
        counter += synapse->border;
        pos++;
    }

    if(outH - counter > 0.01f
            && connection->backwardNextId == UNINIT_STATE_32)
    {
        __global UpdatePosSection* updateSection = &updatePosSections[connection->sourceNeuronSectionId];
        __global UpdatePos* updatePos = &updateSection->positions[connection->sourceNeuronId];
        updatePos->type = 1;
        updatePos->offset = counter + connection->offset;
        segmentSettings->updateSections = 1;
    }

    if(connection->backwardNextId != UNINIT_STATE_32)
    {
        synapseProcessingBackbard(&synapseSections[connection->backwardNextId],
                                  &sectionConnections[connection->backwardNextId],
                                  targetNeuronSection,
                                  neuronSections,
                                  sectionConnections,
                                  synapseSections,
                                  updatePosSections,
                                  segmentSettings,
                                  randomValues);
    }
}

/**
 * @brief process only a single neuron
 */
inline void
processSingleNeuron(const uint sourceNeuronId,
                    const uint sourceNeuronSectionId,
                    __global Neuron* neuron,
                    __global UpdatePosSection* updatePosSections,
                    __global SegmentSettings* segmentSettings)
{
    // handle active-state
    if(neuron->active != 0)
    {
        if(neuron->targetSectionId == UNINIT_STATE_32)
        {
            __global UpdatePos* updatePos = &updatePosSections[sourceNeuronSectionId].positions[sourceNeuronId];
            updatePos->type = 1;
            updatePos->offset = 0.0f;
            segmentSettings->updateSections = 1;
        }
    }
}

/**
 * @brief process output brick
 */
inline void
processNeuronsOfOutputBrick(__global const Brick* brick,
                            __global NeuronSection* neuronSections,
                            __global SectionConnection* sectionConnections,
                            __global SynapseSection* synapseSections,
                            __global UpdatePosSection* updatePosSections,
                            __global float* outputTransfers,
                            __global SegmentSettings* segmentSettings,
                            __global const uint* randomValues)
{
    __global Neuron* neuron = NULL;
    __global NeuronSection* neuronSection = NULL;

    // iterate over all neurons within the brick
    for(uint neuronSectionId = brick->neuronSectionPos + get_global_id(0);
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
        neuronSectionId += get_global_size(0))
    {
        neuronSection = &neuronSections[neuronSectionId];

        if(neuronSection->backwardNextId != UNINIT_STATE_32)
        {
            synapseProcessingBackbard(&synapseSections[neuronSection->backwardNextId],
                                      &sectionConnections[neuronSection->backwardNextId],
                                      neuronSection,
                                      neuronSections,
                                      sectionConnections,
                                      synapseSections,
                                      updatePosSections,
                                      segmentSettings,
                                      randomValues);
        }

        for(uint neuronId = 0;
            neuronId < neuronSection->numberOfNeurons;
            neuronId++)
        {
            neuron = &neuronSection->neurons[neuronId];
            neuron->potential = segmentSettings->potentialOverflow * neuron->input;
            outputTransfers[neuron->targetBorderId] = neuron->potential;
            neuron->input = 0.0f;
        }
    }
}

/**
 * @brief process input brick
 */
inline void
processNeuronsOfInputBrick(__global const Brick* brick,
                           __global NeuronSection* neuronSections,
                           __global float* inputTransfers,
                           __global SynapseSection* synapseSections,
                           __global UpdatePosSection* updatePosSections,
                           __global SegmentSettings* segmentSettings)
{
    __global Neuron* neuron = NULL;
    __global NeuronSection* section = NULL;

    // iterate over all neurons within the brick
    for(uint neuronSectionId = brick->neuronSectionPos + get_global_id(0);
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
        neuronSectionId += get_global_size(0))
    {
        section = &neuronSections[neuronSectionId];
        for(uint neuronId = 0;
            neuronId < section->numberOfNeurons;
            neuronId++)
        {
            neuron = &section->neurons[neuronId];
            neuron->potential = inputTransfers[neuron->targetBorderId];
            neuron->active = neuron->potential > 0.0f;

            processSingleNeuron(neuronId,
                                neuronSectionId,
                                neuron,
                                updatePosSections,
                                segmentSettings);
        }
    }
}

/**
 * @brief process normal internal brick
 */
inline void
processNeuronsOfNormalBrick(__global const Brick* brick,
                            __global NeuronSection* neuronSections,
                            __global SectionConnection* sectionConnections,
                            __global SynapseSection* synapseSections,
                            __global UpdatePosSection* updatePosSections,
                            __global SegmentSettings* segmentSettings,
                            __global const uint* randomValues)
{
    __global Neuron* neuron = NULL;
    __global NeuronSection* section = NULL;

    // iterate over all neurons within the brick
    for(uint neuronSectionId = brick->neuronSectionPos + get_global_id(0);
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
        neuronSectionId += get_global_size(0))
    {
        section = &neuronSections[neuronSectionId];

        if(section->backwardNextId != UNINIT_STATE_32)
        {
            synapseProcessingBackbard(&synapseSections[section->backwardNextId],
                                      &sectionConnections[section->backwardNextId],
                                      section,
                                      neuronSections,
                                      sectionConnections,
                                      synapseSections,
                                      updatePosSections,
                                      segmentSettings,
                                      randomValues);
        }

        for(uint neuronId = 0;
            neuronId < section->numberOfNeurons;
            neuronId++)
        {
            neuron = &section->neurons[neuronId];

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

            processSingleNeuron(neuronId,
                                neuronSectionId,
                                neuron,
                                updatePosSections,
                                segmentSettings);
        }
    }
}

/**
 * @brief process all neurons within a segment
 */
__kernel inline void
prcessCoreSegment(__global Brick* bricks,
                  __global uint* brickOrder,
                  __global NeuronSection* neuronSections,
                  __global SectionConnection* sectionConnections,
                  __global SynapseSection* synapseSections,
                  __global UpdatePosSection* updatePosSections,
                  __global SegmentHeader* segmentHeader,
                  __global SegmentSettings* segmentSettings,
                  __global float* inputTransfers,
                  __global float* outputTransfers,
                  __global const uint* randomValues)
{


    const uint numberOfBricks = segmentHeader->bricks.count;
    for(uint pos = 0; pos < numberOfBricks; pos++)
    {
        const uint brickId = brickOrder[pos];

        __global Brick* brick = &bricks[brickId];
        if(brick->isInputBrick)
        {
            processNeuronsOfInputBrick(brick,
                                       neuronSections,
                                       inputTransfers,
                                       synapseSections,
                                       updatePosSections,
                                       segmentSettings);
        }
        else if(brick->isOutputBrick)
        {
            processNeuronsOfOutputBrick(brick,
                                        neuronSections,
                                        sectionConnections,
                                        synapseSections,
                                        updatePosSections,
                                        outputTransfers,
                                        segmentSettings,
                                        randomValues);
        }
        else
        {
            processNeuronsOfNormalBrick(brick,
                                        neuronSections,
                                        sectionConnections,
                                        synapseSections,
                                        updatePosSections,
                                        segmentSettings,
                                        randomValues);
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

/**
 * @brief backpropagate values of an output-brick
 */
inline void
backpropagateOutput(__global const Brick* brick,
                    __global float* inputTransfers,
                    __global NeuronSection* neuronSections,
                    __global SegmentSettings* segmentSettings)
{
    __global Neuron* neuron = NULL;
    __global NeuronSection* section = NULL;

    // iterate over all neurons within the brick
    for(uint neuronSectionId = brick->neuronSectionPos + get_global_id(0);
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
        neuronSectionId += get_global_size(0))
    {
        section = &neuronSections[neuronSectionId];
        for(uint neuronId = 0;
            neuronId < section->numberOfNeurons;
            neuronId++)
        {
            neuron = &section->neurons[neuronId];
            neuron->delta = inputTransfers[neuron->targetBorderId];
            inputTransfers[neuron->targetBorderId] = 0.0f;
        }
    }
}

/**
 * @brief run backpropagation for a single synapse-section
 */
inline void
backpropagateSection(__global SynapseSection* section,
                     __global SectionConnection* connection,
                     __global Neuron* sourceNeuron,
                     const float outH,
                     __global const Brick* brick,
                     __global NeuronSection* neuronSections,
                     __global SectionConnection* sectionConnections,
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

    if(connection->forwardNextId != UNINIT_STATE_32)
    {
        backpropagateSection(&synapseSections[connection->forwardNextId],
                             &sectionConnections[connection->forwardNextId],
                             sourceNeuron,
                             outH,
                             brick,
                             neuronSections,
                             sectionConnections,
                             synapseSections);
    }
}

/**
 * @brief run back-propagation over all neurons
 */
inline void
backpropagateNeurons(__global const Brick* brick,
                     __global NeuronSection* neuronSections,
                     __global SectionConnection* sectionConnections,
                     __global SynapseSection* synapseSections,
                     __global float* outputTransfers)
{
    __global Neuron* sourceNeuron = NULL;
    __global NeuronSection* neuronSection = NULL;

    // iterate over all neurons within the brick
    for(uint neuronSectionId = brick->neuronSectionPos + get_global_id(0);
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
        neuronSectionId += get_global_size(0))
    {
        neuronSection = &neuronSections[neuronSectionId];
        for(uint neuronId = 0;
            neuronId < neuronSection->numberOfNeurons;
            neuronId++)
        {
            // skip section, if not active
            sourceNeuron = &neuronSection->neurons[neuronId];
            if(sourceNeuron->targetSectionId != UNINIT_STATE_32)
            {
                // set start-values
                sourceNeuron->delta = 0.0f;
                if(sourceNeuron->active)
                {
                    backpropagateSection(&synapseSections[sourceNeuron->targetSectionId],
                                         &sectionConnections[sourceNeuron->targetSectionId],
                                         sourceNeuron,
                                         sourceNeuron->potential,
                                         brick,
                                         neuronSections,
                                         sectionConnections,
                                         synapseSections);

                    sourceNeuron->delta *= 1.4427f * pow(0.5f, sourceNeuron->potential);
                }

                if(brick->isInputBrick) {
                    outputTransfers[sourceNeuron->targetBorderId] = sourceNeuron->delta;
                }
            }
        }
    }
}

/**
 * @brief correct weight of synapses within a segment
 */
__kernel void
reweightCoreSegment(__global Brick* bricks,
                    __global uint* brickOrder,
                    __global NeuronSection* neuronSections,
                    __global SectionConnection* sectionConnections,
                    __global SynapseSection* synapseSections,
                    __global SegmentHeader* segmentHeader,
                    __global SegmentSettings* segmentSettings,
                    __global float* inputTransfers,
                    __global float* outputTransfers)
{
    // run back-propagation over all internal neurons and synapses
    const uint numberOfBricks = segmentHeader->bricks.count;
    for(int pos = numberOfBricks - 1; pos >= 0; pos--)
    {
        const uint brickId = brickOrder[pos];
        __global Brick* brick = &bricks[brickId];
        if(brick->isOutputBrick)
        {
            backpropagateOutput(brick,
                                inputTransfers,
                                neuronSections,
                                segmentSettings);
            barrier(CLK_GLOBAL_MEM_FENCE);

        }
        backpropagateNeurons(brick,
                             neuronSections,
                             sectionConnections,
                             synapseSections,
                             outputTransfers);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
