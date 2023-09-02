/**
 * @file       message_definitions.h
 *
 * @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright  Apache License Version 2.0
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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_MESSAGE_DEFINITIONS_H
#define KITSUNEMIMI_SAKURA_NETWORK_MESSAGE_DEFINITIONS_H

#define DEBUG_MODE false

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include <handler/reply_handler.h>
#include <handler/message_blocker_handler.h>

namespace Kitsunemimi::Sakura
{

#define PROTOCOL_IDENTIFIER 0x6e79616e
#define MESSAGE_DELIMITER 0x70617375
#define MESSAGE_CACHE_SIZE (1024*1024)

// for testing this flag is set to a lower value, so it has to be checked, if already set
#ifndef MAX_SINGLE_MESSAGE_SIZE
#define MAX_SINGLE_MESSAGE_SIZE (128*1024)
#endif

enum types
{
    UNDEFINED_TYPE = 0,
    SESSION_TYPE = 1,
    HEARTBEAT_TYPE = 2,
    ERROR_TYPE = 3,
    STREAM_DATA_TYPE = 4,
    SINGLEBLOCK_DATA_TYPE = 5,
    MULTIBLOCK_DATA_TYPE = 6,
};

enum session_subTypes
{
    SESSION_INIT_START_SUBTYPE = 1,
    SESSION_INIT_REPLY_SUBTYPE = 2,

    SESSION_CLOSE_START_SUBTYPE = 3,
    SESSION_CLOSE_REPLY_SUBTYPE = 4,
};

enum heartbeat_subTypes
{
    HEARTBEAT_START_SUBTYPE = 1,
    HEARTBEAT_REPLY_SUBTYPE = 2,
};

enum error_subTypes
{
    ERROR_FALSE_VERSION_SUBTYPE = 1,
    ERROR_UNKNOWN_SESSION_SUBTYPE = 2,
    ERROR_INVALID_MESSAGE_SUBTYPE = 3,
};

enum stream_data_subTypes
{
    DATA_STREAM_STATIC_SUBTYPE = 1,
    DATA_STREAM_REPLY_SUBTYPE = 2,
};

enum singleblock_data_subTypes
{
    DATA_SINGLE_DATA_SUBTYPE = 1,
    DATA_SINGLE_REPLY_SUBTYPE = 2,
};

enum multiblock_data_subTypes
{
    DATA_MULTI_STATIC_SUBTYPE = 1,
    DATA_MULTI_FINISH_SUBTYPE = 2,
};

//==================================================================================================

/**
 * @brief CommonMessageHeader
 *
 * header-size = 32
 */
struct CommonMessageHeader
{
    const uint32_t protocolIdentifier = PROTOCOL_IDENTIFIER;
    uint8_t version = 0x1;
    uint8_t type = 0;
    uint8_t subType = 0;
    uint8_t flags = 0;   // 0x1 = reply required; 0x2 = is reply;
                         // 0x4 = is request; 0x8 = is response
    uint64_t additionalValues = 0;  // not used at the momment
    uint32_t sessionId = 0;
    uint32_t messageId = 0;
    uint32_t totalMessageSize = 0;
    uint32_t payloadSize = 0;
} __attribute__((packed));

/**
 * @brief CommonMessageFooter
 *
 * footer-size = 8
 */
struct CommonMessageFooter
{
    uint32_t additionalValues = 0;  // not used at the momment
    const uint32_t delimiter = MESSAGE_DELIMITER;
} __attribute__((packed));

//==================================================================================================

/**
 * @brief Session_Init_Start_Message
 */
struct Session_Init_Start_Message
{
    CommonMessageHeader commonHeader;
    uint32_t clientSessionId = 0;
    char sessionIdentifier[64000];
    uint32_t sessionIdentifierSize = 0;
    CommonMessageFooter commonEnd;

    Session_Init_Start_Message()
    {
        commonHeader.type = SESSION_TYPE;
        commonHeader.subType = SESSION_INIT_START_SUBTYPE;
        commonHeader.flags = 0x1;
        commonHeader.totalMessageSize = sizeof(Session_Init_Start_Message);
    }

} __attribute__((packed));

/**
 * @brief Session_Init_Reply_Message
 */
struct Session_Init_Reply_Message
{
    CommonMessageHeader commonHeader;
    uint32_t clientSessionId = 0;
    uint32_t completeSessionId = 0;
    char sessionIdentifier[64000];
    uint32_t sessionIdentifierSize = 0;
    uint8_t padding[4];
    CommonMessageFooter commonEnd;

    Session_Init_Reply_Message()
    {
        commonHeader.type = SESSION_TYPE;
        commonHeader.subType = SESSION_INIT_REPLY_SUBTYPE;
        commonHeader.flags = 0x2;
        commonHeader.totalMessageSize = sizeof(Session_Init_Reply_Message);
    }

} __attribute__((packed));

//==================================================================================================

/**
 * @brief Session_Close_Start_Message
 */
struct Session_Close_Start_Message
{
    CommonMessageHeader commonHeader;
    uint32_t sessionId = 0;
    uint8_t padding[4];
    CommonMessageFooter commonEnd;

    Session_Close_Start_Message()
    {
        commonHeader.type = SESSION_TYPE;
        commonHeader.subType = SESSION_CLOSE_START_SUBTYPE;
        commonHeader.totalMessageSize = sizeof(Session_Close_Start_Message);
    }

} __attribute__((packed));

/**
 * @brief Session_Close_Reply_Message
 */
struct Session_Close_Reply_Message
{
    CommonMessageHeader commonHeader;
    uint32_t sessionId = 0;
    uint8_t padding[4];
    CommonMessageFooter commonEnd;

    Session_Close_Reply_Message()
    {
        commonHeader.type = SESSION_TYPE;
        commonHeader.subType = SESSION_CLOSE_REPLY_SUBTYPE;
        commonHeader.flags = 0x2;
        commonHeader.totalMessageSize = sizeof(Session_Close_Reply_Message);
    }

} __attribute__((packed));

//==================================================================================================

/**
 * @brief Heartbeat_Start_Message
 */
struct Heartbeat_Start_Message
{
    CommonMessageHeader commonHeader;
    CommonMessageFooter commonEnd;

    Heartbeat_Start_Message()
    {
        commonHeader.type = HEARTBEAT_TYPE;
        commonHeader.subType = HEARTBEAT_START_SUBTYPE;
        commonHeader.flags = 0x1;
        commonHeader.totalMessageSize = sizeof(Heartbeat_Start_Message);
    }

} __attribute__((packed));

/**
 * @brief Heartbeat_Reply_Message
 */
struct Heartbeat_Reply_Message
{
    CommonMessageHeader commonHeader;
    CommonMessageFooter commonEnd;

    Heartbeat_Reply_Message()
    {
        commonHeader.type = HEARTBEAT_TYPE;
        commonHeader.subType = HEARTBEAT_REPLY_SUBTYPE;
        commonHeader.flags = 0x2;
        commonHeader.totalMessageSize = sizeof(Heartbeat_Reply_Message);
    }

} __attribute__((packed));

//==================================================================================================

/**
 * @brief Error_FalseVersion_Message
 */
struct Error_FalseVersion_Message
{
    CommonMessageHeader commonHeader;
    uint64_t messageSize = 0;
    char message[MESSAGE_CACHE_SIZE];
    CommonMessageFooter commonEnd;

    Error_FalseVersion_Message()
    {
        commonHeader.type = ERROR_TYPE;
        commonHeader.subType = ERROR_FALSE_VERSION_SUBTYPE;
        commonHeader.totalMessageSize = sizeof(Error_FalseVersion_Message);
    }

} __attribute__((packed));

/**
 * @brief Error_UnknownSession_Message
 */
struct Error_UnknownSession_Message
{
    CommonMessageHeader commonHeader;
    uint64_t messageSize = 0;
    char message[MESSAGE_CACHE_SIZE];
    CommonMessageFooter commonEnd;

    Error_UnknownSession_Message()
    {
        commonHeader.type = ERROR_TYPE;
        commonHeader.subType = ERROR_UNKNOWN_SESSION_SUBTYPE;
        commonHeader.totalMessageSize = sizeof(Error_UnknownSession_Message);
    }

} __attribute__((packed));


/**
 * @brief Error_InvalidMessage_Message
 */
struct Error_InvalidMessage_Message
{
    CommonMessageHeader commonHeader;
    uint64_t messageSize = 0;
    char message[MESSAGE_CACHE_SIZE];
    CommonMessageFooter commonEnd;

    Error_InvalidMessage_Message()
    {
        commonHeader.type = ERROR_TYPE;
        commonHeader.subType = ERROR_INVALID_MESSAGE_SUBTYPE;
        commonHeader.totalMessageSize = sizeof(Error_InvalidMessage_Message);
    }

} __attribute__((packed));

//==================================================================================================

/**
 * @brief Data_Stream_Header
 */
struct Data_Stream_Header
{
    CommonMessageHeader commonHeader;

    Data_Stream_Header()
    {
        commonHeader.type = STREAM_DATA_TYPE;
        commonHeader.subType = DATA_STREAM_STATIC_SUBTYPE;
    }

} __attribute__((packed));

/**
 * @brief Data_StreamReply_Message
 */
struct Data_StreamReply_Message
{
    CommonMessageHeader commonHeader;
    CommonMessageFooter commonEnd;

    Data_StreamReply_Message()
    {
        commonHeader.type = STREAM_DATA_TYPE;
        commonHeader.subType = DATA_STREAM_REPLY_SUBTYPE;
        commonHeader.flags = 0x2;
        commonHeader.totalMessageSize = sizeof(Data_StreamReply_Message);
    }

} __attribute__((packed));

//==================================================================================================

/**
 * @brief Data_SingleBlock_Header
 */
struct Data_SingleBlock_Header
{
    CommonMessageHeader commonHeader;
    uint64_t multiblockId = 0;
    uint64_t blockerId = 0;
    uint8_t padding[4];

    Data_SingleBlock_Header()
    {
        commonHeader.type = SINGLEBLOCK_DATA_TYPE;
        commonHeader.subType = DATA_SINGLE_DATA_SUBTYPE;
        commonHeader.flags = 0x1;
    }

} __attribute__((packed));

/**
 * @brief Data_SingleReply_Message
 */
struct Data_SingleBlockReply_Message
{
    CommonMessageHeader commonHeader;
    CommonMessageFooter commonEnd;

    Data_SingleBlockReply_Message()
    {
        commonHeader.type = SINGLEBLOCK_DATA_TYPE;
        commonHeader.subType = DATA_SINGLE_REPLY_SUBTYPE;
        commonHeader.flags = 0x2;
        commonHeader.totalMessageSize = sizeof(Data_SingleBlockReply_Message);
    }

} __attribute__((packed));

//==================================================================================================

/**
 * @brief Data_MultiBlock_Header
 */
struct Data_MultiBlock_Header
{
    CommonMessageHeader commonHeader;
    uint64_t totalSize = 0;
    uint64_t multiblockId = 0;
    uint32_t totalPartNumber = 0;
    uint32_t partId = 0;

    Data_MultiBlock_Header()
    {
        commonHeader.type = MULTIBLOCK_DATA_TYPE;
        commonHeader.subType = DATA_MULTI_STATIC_SUBTYPE;
    }

} __attribute__((packed));

/**
 * @brief Data_MultiFinish_Message
 */
struct Data_MultiFinish_Message
{
    CommonMessageHeader commonHeader;
    uint64_t multiblockId = 0;
    uint64_t blockerId = 0;
    CommonMessageFooter commonEnd;

    Data_MultiFinish_Message()
    {
        commonHeader.type = MULTIBLOCK_DATA_TYPE;
        commonHeader.subType = DATA_MULTI_FINISH_SUBTYPE;
        commonHeader.totalMessageSize = sizeof(Data_MultiFinish_Message);
    }

} __attribute__((packed));

//==================================================================================================

}

#endif // KITSUNEMIMI_SAKURA_NETWORK_MESSAGE_DEFINITIONS_H
