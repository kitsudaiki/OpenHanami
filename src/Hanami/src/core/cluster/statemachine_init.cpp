/**
 * @file        statemachine_init.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2022 Tobias Anker
 *
 *      Licensed under the Apache License, Version 2.0 (the "License");
 *      you may not use cluster file except in compliance with the License.
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

#include "statemachine_init.h"

#include <core/cluster/states/task_handle_state.h>
#include <core/cluster/states/cycle_finish_state.h>
#include <core/cluster/states/tables/table_interpolation_state.h>
#include <core/cluster/states/tables/table_train_forward_state.h>
#include <core/cluster/states/images/image_identify_state.h>
#include <core/cluster/states/images/image_train_forward_state.h>
#include <core/cluster/states/checkpoints/save_cluster_state.h>
#include <core/cluster/states/checkpoints/restore_cluster_state.h>

#include <core/cluster/cluster.h>

#include <hanami_common/statemachine.h>

/**
 * @brief initialize all possible states of the statemachine
 *
 * @param sm reference to the statemachine, which should be initialized
 */
void
initStates(Hanami::Statemachine &sm)
{
    sm.createNewState(TASK_STATE,                       "Task-handling mode");
    sm.createNewState(TRAIN_STATE,                      "Train-State");
    sm.createNewState(IMAGE_TRAIN_STATE,                "Image-train state");
    sm.createNewState(IMAGE_TRAIN_FORWARD_STATE,        "Image-train state: run");
    sm.createNewState(IMAGE_TRAIN_CYCLE_FINISH_STATE,   "Image-train state: finish-cycle");
    sm.createNewState(TABLE_TRAIN_STATE,                "Table-train state");
    sm.createNewState(TABLE_TRAIN_FORWARD_STATE,        "Table-train state: run");
    sm.createNewState(TABLE_TRAIN_CYCLE_FINISH_STATE,   "Table-train state: finish-cycle");
    sm.createNewState(REQUEST_STATE,                    "Request-State");
    sm.createNewState(IMAGE_REQUEST_STATE,              "Image-request state");
    sm.createNewState(IMAGE_REQUEST_FORWARD_STATE,      "Image-request state: forward-propagation");
    sm.createNewState(IMAGE_REQUEST_CYCLE_FINISH_STATE, "Image-request state: finish-cycle");
    sm.createNewState(TABLE_REQUEST_STATE,              "Table-request state");
    sm.createNewState(TABLE_REQUEST_FORWARD_STATE,      "Table-request state: forward-propagation");
    sm.createNewState(TABLE_REQUEST_CYCLE_FINISH_STATE, "Table-request state: finish-cycle");
    sm.createNewState(CHECKPOINT_STATE,                   "Checkpoint state");
    sm.createNewState(CLUSTER_CHECKPOINT_STATE,           "Cluster-checkpoint state");
    sm.createNewState(CLUSTER_CHECKPOINT_SAVE_STATE,      "Cluster-checkpoint state: save");
    sm.createNewState(CLUSTER_CHECKPOINT_RESTORE_STATE,   "Cluster-checkpoint state: restore");
    sm.createNewState(DIRECT_STATE,                     "Direct mode");
}

/**
 * @brief initialize events, which should be triggered for specific states
 *
 * @param sm reference to the statemachine, which should be initialized
 * @param cluster pointer to the cluster, where the statemachine belongs to
 * @param taskState pointer the the cluster-specific task-handling-state
 */
void
initEvents(Hanami::Statemachine &sm,
           Cluster* cluster,
           TaskHandle_State* taskState)
{
    sm.addEventToState(TASK_STATE,                       taskState);
    sm.addEventToState(IMAGE_TRAIN_FORWARD_STATE,        new ImageTrainForward_State(cluster));
    sm.addEventToState(TABLE_TRAIN_FORWARD_STATE,        new TableTrainForward_State(cluster));
    sm.addEventToState(IMAGE_REQUEST_FORWARD_STATE,      new ImageIdentify_State(cluster));
    sm.addEventToState(TABLE_REQUEST_FORWARD_STATE,      new TableInterpolation_State(cluster));
    sm.addEventToState(IMAGE_TRAIN_CYCLE_FINISH_STATE,   new CycleFinish_State(cluster));
    sm.addEventToState(TABLE_TRAIN_CYCLE_FINISH_STATE,   new CycleFinish_State(cluster));
    sm.addEventToState(IMAGE_REQUEST_CYCLE_FINISH_STATE, new CycleFinish_State(cluster));
    sm.addEventToState(TABLE_REQUEST_CYCLE_FINISH_STATE, new CycleFinish_State(cluster));
    sm.addEventToState(CLUSTER_CHECKPOINT_SAVE_STATE,      new SaveCluster_State(cluster));
    sm.addEventToState(CLUSTER_CHECKPOINT_RESTORE_STATE,   new RestoreCluster_State(cluster));
}

/**
 * @brief initialize child-states
 *
 * @param sm reference to the statemachine, which should be initialized
 */
void
initChildStates(Hanami::Statemachine &sm)
{
    // child states image train
    sm.addChildState(TRAIN_STATE,       IMAGE_TRAIN_STATE);
    sm.addChildState(IMAGE_TRAIN_STATE, IMAGE_TRAIN_FORWARD_STATE);
    sm.addChildState(IMAGE_TRAIN_STATE, IMAGE_TRAIN_CYCLE_FINISH_STATE);

    // child states table train
    sm.addChildState(TRAIN_STATE,       TABLE_TRAIN_STATE);
    sm.addChildState(TABLE_TRAIN_STATE, TABLE_TRAIN_FORWARD_STATE);
    sm.addChildState(TABLE_TRAIN_STATE, TABLE_TRAIN_CYCLE_FINISH_STATE);

    // child states image request
    sm.addChildState(REQUEST_STATE,       IMAGE_REQUEST_STATE);
    sm.addChildState(IMAGE_REQUEST_STATE, IMAGE_REQUEST_FORWARD_STATE);
    sm.addChildState(IMAGE_REQUEST_STATE, IMAGE_REQUEST_CYCLE_FINISH_STATE);

    // child states table request
    sm.addChildState(REQUEST_STATE,       TABLE_REQUEST_STATE);
    sm.addChildState(TABLE_REQUEST_STATE, TABLE_REQUEST_FORWARD_STATE);
    sm.addChildState(TABLE_REQUEST_STATE, TABLE_REQUEST_CYCLE_FINISH_STATE);

    // child states checkpoint
    sm.addChildState(CHECKPOINT_STATE,         CLUSTER_CHECKPOINT_STATE);
    sm.addChildState(CLUSTER_CHECKPOINT_STATE, CLUSTER_CHECKPOINT_SAVE_STATE);
    sm.addChildState(CLUSTER_CHECKPOINT_STATE, CLUSTER_CHECKPOINT_RESTORE_STATE);
}

/**
 * @brief set initial
 *
 * @param sm reference to the statemachine, which should be initialized
 */
void
initInitialChildStates(Hanami::Statemachine &sm)
{
    sm.setInitialChildState(IMAGE_TRAIN_STATE,   IMAGE_TRAIN_FORWARD_STATE);
    sm.setInitialChildState(TABLE_TRAIN_STATE,   TABLE_TRAIN_FORWARD_STATE);
    sm.setInitialChildState(IMAGE_REQUEST_STATE, IMAGE_REQUEST_FORWARD_STATE);
    sm.setInitialChildState(TABLE_REQUEST_STATE, TABLE_REQUEST_FORWARD_STATE);
}

/**
 * @brief initialize transitions between states
 *
 * @param sm reference to the statemachine, which should be initialized
 */
void
initTransitions(Hanami::Statemachine &sm)
{
    // transtions train init
    sm.addTransition(TASK_STATE,  TRAIN, TRAIN_STATE);
    sm.addTransition(TRAIN_STATE, IMAGE, IMAGE_TRAIN_STATE);
    sm.addTransition(TRAIN_STATE, TABLE, TABLE_TRAIN_STATE);

    // transitions request init
    sm.addTransition(TASK_STATE,    REQUEST, REQUEST_STATE);
    sm.addTransition(REQUEST_STATE, IMAGE,   IMAGE_REQUEST_STATE);
    sm.addTransition(REQUEST_STATE, TABLE,   TABLE_REQUEST_STATE);

    // transitions checkpoint init
    sm.addTransition(TASK_STATE,             CHECKPOINT, CHECKPOINT_STATE);
    sm.addTransition(CHECKPOINT_STATE,         CLUSTER,  CLUSTER_CHECKPOINT_STATE);
    sm.addTransition(CLUSTER_CHECKPOINT_STATE, SAVE,     CLUSTER_CHECKPOINT_SAVE_STATE);
    sm.addTransition(CLUSTER_CHECKPOINT_STATE, RESTORE,  CLUSTER_CHECKPOINT_RESTORE_STATE);

    // trainsition train-internal
    sm.addTransition(IMAGE_TRAIN_FORWARD_STATE,      NEXT, IMAGE_TRAIN_CYCLE_FINISH_STATE );
    sm.addTransition(IMAGE_TRAIN_CYCLE_FINISH_STATE, NEXT, IMAGE_TRAIN_FORWARD_STATE      );
    sm.addTransition(TABLE_TRAIN_FORWARD_STATE,      NEXT, TABLE_TRAIN_CYCLE_FINISH_STATE );
    sm.addTransition(TABLE_TRAIN_CYCLE_FINISH_STATE, NEXT, TABLE_TRAIN_FORWARD_STATE      );

    // trainsition request-internal
    sm.addTransition(IMAGE_REQUEST_FORWARD_STATE,      NEXT, IMAGE_REQUEST_CYCLE_FINISH_STATE );
    sm.addTransition(IMAGE_REQUEST_CYCLE_FINISH_STATE, NEXT, IMAGE_REQUEST_FORWARD_STATE      );
    sm.addTransition(TABLE_REQUEST_FORWARD_STATE,      NEXT, TABLE_REQUEST_CYCLE_FINISH_STATE );
    sm.addTransition(TABLE_REQUEST_CYCLE_FINISH_STATE, NEXT, TABLE_REQUEST_FORWARD_STATE      );

    // transition finish back to task-state
    sm.addTransition(TRAIN_STATE,                    FINISH_TASK, TASK_STATE);
    sm.addTransition(REQUEST_STATE,                  FINISH_TASK, TASK_STATE);
    sm.addTransition(CHECKPOINT_STATE,                 FINISH_TASK, TASK_STATE);
    sm.addTransition(CLUSTER_CHECKPOINT_SAVE_STATE,    FINISH_TASK, TASK_STATE);
    sm.addTransition(CLUSTER_CHECKPOINT_RESTORE_STATE, FINISH_TASK, TASK_STATE);

    // special transition to tigger the task-state again
    sm.addTransition(TASK_STATE, PROCESS_TASK, TASK_STATE);

    // mode-switches
    sm.addTransition(TASK_STATE, SWITCH_TO_DIRECT_MODE, DIRECT_STATE);
    sm.addTransition(DIRECT_STATE, SWITCH_TO_TASK_MODE, TASK_STATE);
}

/**
 * @brief initialize statemachine of the cluster
 *
 * @param sm reference to the statemachine, which should be initialized
 * @param cluster pointer to the cluster, where the statemachine belongs to
 * @param taskState pointer the the cluster-specific task-handling-state
 */
void
initStatemachine(Hanami::Statemachine &sm,
                 Cluster* cluster,
                 TaskHandle_State* taskState)
{
    initStates(sm);
    initEvents(sm, cluster, taskState);
    initChildStates(sm);
    initInitialChildStates(sm);
    initTransitions(sm);

    // set initial state for the state-machine
    sm.setCurrentState(TASK_STATE);
}
