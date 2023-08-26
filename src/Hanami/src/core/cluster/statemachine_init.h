/**
 * @file        statemachine_init.h
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

#ifndef HANAMI_STATEMACHINE_INIT_H
#define HANAMI_STATEMACHINE_INIT_H

class TaskHandle_State;
class CycleFinish_State;
class TableInterpolation_State;
class TableLearnBackward_State;
class TableLearnForward_State;
class ImageIdentify_State;
class ImageLearnBackward_State;
class ImageLearnForward_State;
class Cluster;

namespace Kitsunemimi {
class EventQueue;
class Statemachine;
}

enum ClusterStates
{
    TASK_STATE = 0,
    LEARN_STATE = 1,
        IMAGE_LEARN_STATE = 2,
            IMAGE_LEARN_FORWARD_STATE = 3,
            IMAGE_LEARN_CYCLE_FINISH_STATE = 5,
        TABLE_LEARN_STATE = 6,
            TABLE_LEARN_FORWARD_STATE = 7,
            TABLE_LEARN_CYCLE_FINISH_STATE = 9,
    REQUEST_STATE = 10,
        IMAGE_REQUEST_STATE = 11,
            IMAGE_REQUEST_FORWARD_STATE = 12,
            IMAGE_REQUEST_CYCLE_FINISH_STATE = 13,
        TABLE_REQUEST_STATE = 14,
            TABLE_REQUEST_FORWARD_STATE = 15,
            TABLE_REQUEST_CYCLE_FINISH_STATE = 16,
    SNAPSHOT_STATE = 17,
        CLUSTER_SNAPSHOT_STATE = 18,
            CLUSTER_SNAPSHOT_SAVE_STATE = 19,
            CLUSTER_SNAPSHOT_RESTORE_STATE = 20,
    DIRECT_STATE = 21,
};

enum ClusterTransitions
{
    LEARN = 100,
    REQUEST = 101,
    SNAPSHOT = 102,
    IMAGE = 103,
    TABLE = 104,
    CLUSTER = 105,
    SAVE = 106,
    RESTORE = 107,
    NEXT = 108,
    FINISH_TASK = 109,
    PROCESS_TASK = 110,
    SWITCH_TO_DIRECT_MODE = 111,
    SWITCH_TO_TASK_MODE = 112,
};

void initStatemachine(Kitsunemimi::Statemachine &sm,
                      Cluster* cluster,
                      TaskHandle_State* taskState);

#endif // HANAMI_STATEMACHINE_INIT_H
