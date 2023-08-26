 /* Apache License Version 2.0
 *
 *      Copyright 2021 Tobias Anker
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

enum ShioriMessageTypes
{
    SHIORI_UNDEFINED_MESSAGE_TYPE = 0,
    SHIORI_DATASET_REQUEST_MESSAGE_TYPE = 1,
    SHIORI_RESULT_PUSH_MESSAGE_TYPE = 2,
    SHIORI_AUDIT_LOG_MESSAGE_TYPE = 3,
    SHIORI_ERROR_LOG_MESSAGE_TYPE = 4,
    SHIORI_CLUSTER_CHECKPOINT_PUSH_MESSAGE_TYPE = 5,
    SHIORI_CLUSTER_CHECKPOINT_PULL_MESSAGE_TYPE = 6,
};


enum AzukiMessageTypes
{
    AZUKI_UNDEFINED_MESSAGE_TYPE = 0,
    AZUKI_SPEED_SET_MESSAGE_TYPE = 1,
};