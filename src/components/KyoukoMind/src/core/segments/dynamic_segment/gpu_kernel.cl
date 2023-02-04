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
#define NEURONS_PER_NEURONSECTION 62
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

typedef struct DynamicNeuron_struct
{
    float input;
    float border;
    float potential;
    float delta;

    uchar refractionTime;
    uchar active;
    uchar padding[2];

    uint id;

    uint targetBorderId;
    uint targetSectionId;

    // total size: 32 Byte
} DynamicNeuron;

//==================================================================================================

typedef struct Synapse_struct
{
    float weight;
    float border;
    ushort targetNeuronId;
    char activeCounter;
    uchar active;
    uchar padding[4];
    // total size: 16 Byte
} Synapse;

//==================================================================================================

typedef struct Brick_struct
{
    // common
    uint brickId;
    bool isOutputBrick;
    bool isTransactionBrick;
    bool isInputBrick;
    uchar padding1[13];
    uint neuronSectionPos;

    Position brickPos;
    uint neighbors[12];

    uint possibleTargetNeuronBrickIds[1000];
    uint numberOfNeurons;
    uint numberOfNeuronSections;

    // total size: 4096 Bytes
} Brick;

//==================================================================================================

typedef struct NeuronSection_struct
{
    DynamicNeuron neurons[NEURONS_PER_NEURONSECTION];
    uint numberOfNeurons;
    uint id;
    uint brickId;
    uint backwardNextId;
    uchar padding[48];
    // total size: 2048 Byte
} NeuronSection;

//==================================================================================================

typedef struct SynapseSection_struct
{
    uchar active;
    uchar padding[3];
    uint randomPos;

    uint targetNeuronSectionId;
    uint brickId;
    uchar padding2[8];
    uint forwardNext;
    uint backwardNext;

    Synapse synapses[SYNAPSES_PER_SYNAPSESECTION];

    // total size: 512 Byte
} SynapseSection;

//==================================================================================================

typedef struct UpdatePos_struct
{
    uint type;
    uint forwardNewId;
    uint randomPos;
    uint targetNeuronSectionId;
} UpdatePos;

//==================================================================================================

typedef struct UpdatePosSection_struct
{
    UpdatePos positions[NEURONS_PER_NEURONSECTION];
    uint numberOfPositions;
    uint backwardNewId;
    uchar padding[24];

   // total size: 512 Byte
} UpdatePosSection;

//==================================================================================================

typedef struct DynamicSegmentSettings_struct
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
} DynamicSegmentSettings;

//==================================================================================================

inline void
initNewSection(const uint position,
               __global SynapseSection* synapseSections,
               __global UpdatePos* updatePos,
               __global const uint* randomValues)
{
    __global SynapseSection* targetSection = &synapseSections[position];
    targetSection->active = 1;
    targetSection->randomPos = updatePos->randomPos;
    targetSection->targetNeuronSectionId = updatePos->targetNeuronSectionId;
}

/**
 * @brief initialize a new specific synapse
 *
 * @param section current processed synapse-section
 * @param synapse new synapse, which has to be initialized
 * @param bricks array of all bricks
 * @param sourceNeuron source-neuron, who triggered the section
 * @param segmentSettings settings of the section
 * @param remainingWeight weight of which to cut of a part for the new synapse
 */
inline void
createNewSynapse(__global SynapseSection* section,
                 __global Synapse* synapse,
                 __global const NeuronSection* neuronSections,
                 __global const DynamicSegmentSettings* segmentSettings,
                 const float remainingWeight,
                 const float outH,
                 __global const uint* randomValues)
{
    const float randMax = (float)(RAND_MAX);
    const float maxWeight = outH / (float)(segmentSettings->synapseSegmentation);
    uint signRand = 0;
    const float sigNeg = 0.5f;

    // set activation-border
    section->randomPos = (section->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    float newWeight = maxWeight * ((float)(randomValues[section->randomPos]) / randMax);
    synapse->border = (float)(remainingWeight < newWeight) * remainingWeight
                      + (float)(remainingWeight >= newWeight) * newWeight;

    // set target neuron
    section->randomPos = (section->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->targetNeuronId = (ushort)(randomValues[section->randomPos]
                              % neuronSections[section->targetNeuronSectionId].numberOfNeurons);

    section->randomPos = (section->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    synapse->weight = ((float)(randomValues[section->randomPos]) / randMax) / 10.0f;

    // update weight with sign
    section->randomPos = (section->randomPos + 1) % NUMBER_OF_RAND_VALUES;
    signRand = randomValues[section->randomPos] % 1000;
    synapse->weight *= (float)(1.0f - (1000.0f * sigNeg > signRand) * 2);
    synapse->active = 0;

    synapse->activeCounter = 1;
}

inline void
synapseProcessingBackward(const uint neuronSectionId,
                          __global NeuronSection* neuronSection,
                          __global SynapseSection* section,
                          __global SynapseSection* synapseSections,
                          __global UpdatePosSection* updatePosSections)
{
    uint pos = 0;
    __global Synapse* synapse = NULL;
    __global DynamicNeuron* targetNeuron = NULL;
    uchar active = 0;

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION)
    {
        synapse = &section->synapses[pos];
        neuronSection->neurons[synapse->targetNeuronId].input += ((float)(synapse->active)) * synapse->weight;
        synapse->active = 0;
        pos++;
    }

    if(section->backwardNext == UNINIT_STATE_32)
    {
        __global UpdatePosSection* updatePosSection = &updatePosSections[neuronSectionId];
        section->backwardNext = updatePosSection->backwardNewId;
        updatePosSection->backwardNewId = UNINIT_STATE_32;

        if(section->backwardNext != UNINIT_STATE_32)
        {
            __global SynapseSection* targetSection = &synapseSections[section->backwardNext];
            if(targetSection->active != 1) 
            {
                targetSection->active = 1;
                targetSection->randomPos = neuronSectionId % NUMBER_OF_RAND_VALUES;
                targetSection->targetNeuronSectionId = neuronSectionId;
            }
        }
    }

    if(section->backwardNext != UNINIT_STATE_32)
    {
        synapseProcessingBackward(neuronSectionId,
                                  neuronSection,
                                  &synapseSections[section->backwardNext],
                                  synapseSections,
                                  updatePosSections);
    }
}

inline void
processSingleSectionBackward(const uint neuronSectionId,
                             __global NeuronSection* neuronSections,
                             __global SynapseSection* synapseSections,
                             __global UpdatePosSection* updatePosSections)
{
    __global NeuronSection* section = &neuronSections[neuronSectionId];

    if(section->backwardNextId == UNINIT_STATE_32)
    {
        __global UpdatePosSection* updatePosSection = &updatePosSections[neuronSectionId];
        section->backwardNextId = updatePosSection->backwardNewId;
        updatePosSection->backwardNewId = UNINIT_STATE_32;

        if(section->backwardNextId != UNINIT_STATE_32)
        {
            __global SynapseSection* targetSection = &synapseSections[section->backwardNextId];
            if(targetSection->active != 1) 
            {
                targetSection->active = 1;
                targetSection->randomPos = neuronSectionId % NUMBER_OF_RAND_VALUES;
                targetSection->targetNeuronSectionId = neuronSectionId;
            }
        }
    }

    if(section->backwardNextId != UNINIT_STATE_32)
    {
        synapseProcessingBackward(neuronSectionId,
                                  section,
                                  &synapseSections[section->backwardNextId],
                                  synapseSections,
                                  updatePosSections);
    }
}


/**
 * @brief process synapse-section
 *
 * @param section current processed synapse-section
 * @param segment refernece to the processed segment
 * @param sourceNeuron source-neuron, who triggered the section
 * @param netH wight-value, which comes into the section
 * @param outH multiplicator
 */
inline void
synapseProcessing(const uint neuronId,
                  const uint neuronSectionId,
                  __global SynapseSection* section,
                  __global const DynamicNeuron* sourceNeuron,
                  __global NeuronSection* neuronSections,
                  __global SynapseSection* synapseSections,
                  __global UpdatePosSection* updatePosSections,
                  __global DynamicSegmentSettings* dynamicSegmentSettings,
                  float netH,
                  const float outH,
                  __global const uint* randomValues)
{
    uint pos = 0;
    __global Synapse* synapse = NULL;
    __global DynamicNeuron* targetNeuron = NULL;
    uchar active = 0;

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && netH > 0.0f)
    {
        synapse = &section->synapses[pos];

        // create new synapse if necesarry and learning is active
        if(synapse->targetNeuronId == UNINIT_STATE_16)
        {
            createNewSynapse(section,
                             synapse,
                             neuronSections,
                             dynamicSegmentSettings,
                             netH,
                             outH,
                             randomValues);
        }

        // update target-neuron
        //targetNeuron = &(neuronSections[section->targetNeuronSectionId].neurons[synapse->targetNeuronId]);
        //targetNeuron->input += synapse->weight;

        // update active-counter
        active = (synapse->weight > 0) == (targetNeuron->potential > targetNeuron->border);
        synapse->activeCounter += active * (uchar)(synapse->activeCounter < 126);
        synapse->active = 1;

        // update loop-counter
        netH -= synapse->border;
        pos++;
    }

    if(netH > 0.01f)
    {
        if(section->forwardNext == UNINIT_STATE_32)
        {
            __global UpdatePos* updatePos = &updatePosSections[neuronSectionId].positions[neuronId];
            section->forwardNext= updatePos->forwardNewId;
            updatePos->forwardNewId = UNINIT_STATE_32;
            updatePos->type = section->forwardNext == UNINIT_STATE_32;

            if(section->forwardNext != UNINIT_STATE_32)
            {
                __global SynapseSection* targetSection = &synapseSections[section->forwardNext];
                targetSection->active = 1;
                targetSection->randomPos = updatePos->randomPos;
                targetSection->targetNeuronSectionId = updatePos->targetNeuronSectionId;
            }
        }

        if(section->forwardNext != UNINIT_STATE_32)
        {

                synapseProcessing(neuronId,
                                  neuronSectionId,
                                  &synapseSections[section->forwardNext],
                                  sourceNeuron,
                                  neuronSections,
                                  synapseSections,
                                  updatePosSections,
                                  dynamicSegmentSettings,
                                  netH,
                                  outH,
                                  randomValues);
        }
    }
}

/**
 * @brief process only a single neuron
 *
 * @param neuron pointer to neuron to process
 * @param segment segment where the neuron belongs to
 */
inline void
processSingleNeuron(const uint neuronId,
                    const uint neuronSectionId,
                    __global DynamicNeuron* neuron,
                    __global NeuronSection* neuronSections,
                    __global SynapseSection* synapseSections,
                    __global UpdatePosSection* updatePosSections,
                    __global DynamicSegmentSettings* dynamicSegmentSettings,
                    __global const uint* randomValues)
{
    // handle active-state
    if(neuron->active != 0)
    {
        if(neuron->targetSectionId == UNINIT_STATE_32)
        {
            __global UpdatePos* updatePos = &updatePosSections[neuronSectionId].positions[neuronId];
            neuron->targetSectionId = updatePos->forwardNewId;
            updatePos->forwardNewId = UNINIT_STATE_32;
            updatePos->type = neuron->targetSectionId == UNINIT_STATE_32;

            if(neuron->targetSectionId != UNINIT_STATE_32)
            {
                __global SynapseSection* targetSection = &synapseSections[neuron->targetSectionId];
                targetSection->active = 1;
                targetSection->randomPos = updatePos->randomPos;
                targetSection->targetNeuronSectionId = updatePos->targetNeuronSectionId;
            }
        }

        if(neuron->targetSectionId != UNINIT_STATE_32)
        {
            synapseProcessing(neuronId,
                              neuronSectionId,
                              &synapseSections[neuron->targetSectionId],
                              neuron,
                              neuronSections,
                              synapseSections,
                              updatePosSections,
                              dynamicSegmentSettings,
                              neuron->potential,
                              neuron->potential,
                              randomValues);
        }
    }
}

/**
 * @brief processNeuron
 * @param neuron
 * @param segment
 */
inline void
processNeuron(__global DynamicNeuron* neuron,
              __global DynamicSegmentSettings* dynamicSegmentSettings)
{
    neuron->potential /= dynamicSegmentSettings->neuronCooldown;
    neuron->refractionTime = neuron->refractionTime >> 1;

    if(neuron->refractionTime == 0)
    {
        neuron->potential = dynamicSegmentSettings->potentialOverflow * neuron->input;
        neuron->refractionTime = dynamicSegmentSettings->refractionTime;
    }

    // update neuron
    neuron->potential -= neuron->border;
    neuron->active = neuron->potential > 0.0f;
    neuron->input = 0.0f;
    neuron->potential = log2(neuron->potential + 1.0f);
}

/**
 * @brief reset neurons of a output brick
 *
 * @param brick pointer to the brick
 * @param segment segment where the brick belongs to
 */
inline void
processNeuronsOfOutputBrick(__global const Brick* brick,
                            __global NeuronSection* neuronSections,
                            __global float* outputTransfers,
                            __global SynapseSection* synapseSections,
                            __global UpdatePosSection* updatePosSections,
                            __global DynamicSegmentSettings* dynamicSegmentSettings)
{
    __global DynamicNeuron* neuron = NULL;
    __global NeuronSection* section = NULL;

    // iterate over all neurons within the brick
    for(uint neuronSectionId = brick->neuronSectionPos + get_global_id(0);
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
        neuronSectionId += get_global_size(0))
    {
        processSingleSectionBackward(neuronSectionId,
                                     neuronSections,
                                     synapseSections,
                                     updatePosSections);
        barrier(CLK_GLOBAL_MEM_FENCE);

        section = &neuronSections[neuronSectionId];
        for(uint neuronId = 0;
            neuronId < section->numberOfNeurons;
            neuronId++)
        {
            neuron = &section->neurons[neuronId];
            neuron->potential = dynamicSegmentSettings->potentialOverflow * neuron->input;
            outputTransfers[neuron->targetBorderId] = neuron->potential;
            neuron->input = 0.0f;
        }
    }
}

/**
 * @brief reset neurons of a input brick
 *
 * @param brick pointer to the brick
 * @param segment segment where the brick belongs to
 */
inline void
processNeuronsOfInputBrick(__global const Brick* brick,
                           __global NeuronSection* neuronSections,
                           __global float* inputTransfers,
                           __global SynapseSection* synapseSections,
                           __global UpdatePosSection* updatePosSections,
                           __global DynamicSegmentSettings* dynamicSegmentSettings,
                           __global const uint* randomValues)
{
    __global DynamicNeuron* neuron = NULL;
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
                                neuronSections,
                                synapseSections,
                                updatePosSections,
                                dynamicSegmentSettings,
                                randomValues);
        }
    }
}

/**
 * @brief reset neurons of a normal brick
 *
 * @param brick pointer to the brick
 * @param segment segment where the brick belongs to
 */
inline void
processNeuronsOfNormalBrick(__global const Brick* brick,
                            __global NeuronSection* neuronSections,
                            __global SynapseSection* synapseSections,
                            __global UpdatePosSection* updatePosSections,
                            __global DynamicSegmentSettings* dynamicSegmentSettings,
                            __global const uint* randomValues)
{
    __global DynamicNeuron* neuron = NULL;
    __global NeuronSection* section = NULL;

    // iterate over all neurons within the brick
    for(uint neuronSectionId = brick->neuronSectionPos + get_global_id(0);
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
        neuronSectionId += get_global_size(0))
    {
        processSingleSectionBackward(neuronSectionId,
                                     neuronSections,
                                     synapseSections,
                                     updatePosSections);
        barrier(CLK_GLOBAL_MEM_FENCE);

        section = &neuronSections[neuronSectionId];
        for(uint neuronId = 0;
            neuronId < section->numberOfNeurons;
            neuronId++)
        {
            neuron = &section->neurons[neuronId];
            processNeuron(neuron, dynamicSegmentSettings);
            processSingleNeuron(neuronId,
                                neuronSectionId,
                                neuron,
                                neuronSections,
                                synapseSections,
                                updatePosSections,
                                dynamicSegmentSettings,
                                randomValues);
        }
    }
}

/**
 * @brief process all neurons within a specific brick and also all synapse-sections,
 *        which are connected to an active neuron
 *
 * @param segment segment to process
 */
__kernel void
prcessDynamicSegment(__global Brick* bricks,
                     __global uint* brickOrder,
                     __global NeuronSection* neuronSections,
                     __global SynapseSection* synapseSections,
                     __global UpdatePosSection* updatePosSections,
                     __global SegmentHeader* segmentHeader,
                     __global DynamicSegmentSettings* dynamicSegmentSettings,
                     __global float* inputTransfers,
                     __global float* outputTransfers,
                     __global uint* randomValues)
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
                                       dynamicSegmentSettings,
                                       randomValues);
        }
        else if(brick->isOutputBrick)
        {
            processNeuronsOfOutputBrick(brick,
                                        neuronSections,
                                        outputTransfers,
                                        synapseSections,
                                        updatePosSections,
                                        dynamicSegmentSettings);
        }
        else
        {
            processNeuronsOfNormalBrick(brick,
                                        neuronSections,
                                        synapseSections,
                                        updatePosSections,
                                        dynamicSegmentSettings,
                                        randomValues);
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}


/**
 * @brief backpropagate values of an output-brick
 *
 * @param brick brick to process
 * @param segment segment where the brick belongs to
 */
inline void
backpropagateOutput(__global const Brick* brick,
                    __global float* inputTransfers,
                    __global NeuronSection* neuronSections,
                    __global DynamicSegmentSettings* dynamicSegmentSettings)
{
    __global DynamicNeuron* neuron = NULL;
    __global NeuronSection* section = NULL;

    for(uint neuronSectionId = brick->neuronSectionPos + get_group_id(0);
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
        neuronSectionId += get_global_size(0))
    {
        section = &neuronSections[neuronSectionId];
        for(uint neuronId = get_local_id(0);
            neuronId < section->numberOfNeurons;
            neuronId += get_local_size(0))
        {
            neuron = &section->neurons[neuronId];
            neuron->delta = inputTransfers[neuron->targetBorderId];
            inputTransfers[neuron->targetBorderId] = 0.0f;
        }
    }
}

/**
 * @brief run backpropagation for a single synapse-section
 *
 * @param section pointer to section to process
 * @param sourceNeuron pointer to the neuron, who triggered the section
 * @param netH neuron-potential
 * @param outH output-multiplicator
 * @param brick brick where the seciton is located
 * @param segment segment where section belongs to
 */
inline void
backpropagateSection(__global SynapseSection* section,
                     __global DynamicNeuron* sourceNeuron,
                     float netH,
                     __global const Brick* brick,
                     __global NeuronSection* neuronSections,
                     __global SynapseSection* synapseSections)
{
    __global Synapse* synapse = NULL;
    __global DynamicNeuron* targetNeuron = NULL;
    __global NeuronSection* neuronSection = &neuronSections[section->targetNeuronSectionId];
    float learnValue = 0.2f;
    ushort pos = 0;

    // iterate over all synapses in the section
    while(pos < SYNAPSES_PER_SYNAPSESECTION
          && netH > 0.0f)
    {
        // break look, if no more synapses to process
        synapse = &section->synapses[pos];

        // update weight
        learnValue = (float)(126 - synapse->activeCounter) * 0.0002f;
        learnValue += 0.05f;
        targetNeuron = &neuronSection->neurons[synapse->targetNeuronId];
        sourceNeuron->delta += targetNeuron->delta * synapse->weight;
        synapse->weight -= learnValue * targetNeuron->delta;

        netH -= synapse->border;
        pos++;
    }

    if(section->forwardNext != UNINIT_STATE_32
            && netH > 0.01f)
    {
        backpropagateSection(&synapseSections[section->forwardNext],
                             sourceNeuron,
                             netH,
                             brick,
                             neuronSections,
                             synapseSections);
    }
}

/**
 * @brief run back-propagation over the hidden neurons
 *
 * @param brick pointer to current brick
 * @param segment pointer to currect segment to process, which contains the brick
 */
inline void
backpropagateNeurons(__global const Brick* brick,
                     __global NeuronSection* neuronSections,
                     __global SynapseSection* synapseSections,
                     __global UpdatePosSection* updatePosSections,
                     __global float* outputTransfers)
{
    __global DynamicNeuron* sourceNeuron = NULL;
    __global NeuronSection* neuronSection = NULL;
    __global UpdatePosSection* updatePosSection = NULL;

    // iterate over all neurons within the brick
    for(uint neuronSectionId = brick->neuronSectionPos + get_group_id(0);
        neuronSectionId < brick->numberOfNeuronSections + brick->neuronSectionPos;
        neuronSectionId += get_global_size(0))
    {
        neuronSection = &neuronSections[neuronSectionId];
        updatePosSection = &updatePosSections[neuronSectionId];
        for(uint neuronId = get_local_id(0);
            neuronId < neuronSection->numberOfNeurons;
            neuronId += get_local_size(0))
        {
            // skip section, if not active
            sourceNeuron = &neuronSection->neurons[neuronId];
            //UpdatePos* updatePos = &updatePosSection->positions[neuronId];
            if(sourceNeuron->targetSectionId != UNINIT_STATE_32)
            {
                sourceNeuron->delta = 0.0f;

                // set start-values
                if(sourceNeuron->active)
                {
                    backpropagateSection(&synapseSections[sourceNeuron->targetSectionId],
                                         sourceNeuron,
                                         sourceNeuron->potential,
                                         brick,
                                         neuronSections,
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
 * @brief correct wight of synapses within
 *
 * @param segment segment to process
 */
__kernel void
rewightDynamicSegment(__global Brick* bricks,
                      __global uint* brickOrder,
                      __global NeuronSection* neuronSections,
                      __global SynapseSection* synapseSections,
                      __global UpdatePosSection* updatePosSections,
                      __global SegmentHeader* segmentHeader,
                      __global DynamicSegmentSettings* dynamicSegmentSettings,
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
                                dynamicSegmentSettings);
            barrier(CLK_GLOBAL_MEM_FENCE);
        }

        backpropagateNeurons(brick,
                             neuronSections,
                             synapseSections,
                             updatePosSections,
                             outputTransfers);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
