/**
 * @file        defines.h
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

#define UNINTI_POINT_32 0x0FFFFFFF

// const predefined values
#define UNINIT_STATE_64 0xFFFFFFFFFFFFFFFF
#define UNINIT_STATE_32 0xFFFFFFFF
#define UNINIT_STATE_24 0xFFFFFF
#define UNINIT_STATE_16 0xFFFF
#define UNINIT_STATE_8 0xFF

#define UNINTI_POINT_32 0x0FFFFFFF

#define TRANSFER_SEGMENT_SIZE (128 * 1024)

// regex
#define UUID_REGEX "[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}"
#define ID_REGEX "[a-zA-Z][a-zA-Z_0-9]*"
#define ID_EXT_REGEX "[a-zA-Z][a-zA-Z_0-9@]*"
#define NAME_REGEX "[a-zA-Z][a-zA-Z_0-9 ]*"
#define INT_VALUE_REGEX "^-?([0-9]+)$"
#define FLOAT_VALUE_REGEX "^-?([0-9]+)\\.([0-9]+)$"
#define IPV4_REGEX "\\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\\.|$)){4}\\b"
