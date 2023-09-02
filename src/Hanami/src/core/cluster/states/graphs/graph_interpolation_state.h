/**
 * @file        graph_interpolation_state.h
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

#ifndef HANAMI_GRAPHINTERPOLATION_STATE_H
#define HANAMI_GRAPHINTERPOLATION_STATE_H

#include <hanami_common/threading/event.h>

class Cluster;

class GraphInterpolation_State
        : public Hanami::Event
{
public:
    GraphInterpolation_State(Cluster* cluster);
    ~GraphInterpolation_State();

    bool processEvent();

private:
    Cluster* m_cluster = nullptr;
};

#endif // HANAMI_GRAPHINTERPOLATION_STATE_H
