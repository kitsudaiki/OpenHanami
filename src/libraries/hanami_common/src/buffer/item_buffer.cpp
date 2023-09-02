/**
 *  @file       item_buffer.cpp
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  Apache License Version 2.0
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

#include <hanami_common/buffer/item_buffer.h>

namespace Hanami
{

struct EmptyPlaceHolder
{
    uint8_t status = ItemBuffer::DELETED_SECTION;
    uint64_t bytePositionOfNextEmptyBlock = ITEM_BUFFER_UNDEFINE_POS;
} __attribute__((packed));

/**
 * @brief constructor
 */
ItemBuffer::ItemBuffer() {}

/**
 * @brief destructor
 */
ItemBuffer::~ItemBuffer() {}

/**
 * @brief init basically a simple buffer without items.
 *
 * @param staticSize number of bytes to allocate
 *
 * @return true, if successful, else false
 */
bool
ItemBuffer::initBuffer(const uint64_t staticSize)
{
    // allocate memory
    const bool ret = initDataBlocks(0, 0, staticSize);
    if(ret == false) {
        return false;
    }

    return true;
}

/**
 * @brief initialize a new item-buffer based on the payload of an old one
 *
 * @param data pointer to the data to import
 * @param dataSize number of bytes of data
 *
 * @return true, if successful, else false if input is invalid
 */
bool
ItemBuffer::initBuffer(const void* data, const uint64_t dataSize)
{
    // precheck
    if(dataSize == 0
            || data == nullptr)
    {
        return false;
    }

    // allocate blocks in buffer and fill with old data
    Hanami::allocateBlocks_DataBuffer(buffer, calcBytesToBlocks(dataSize));
    buffer.usedBufferSize = dataSize;
    memcpy(buffer.data, data, dataSize);

    // init pointer
    metaData = static_cast<MetaData*>(buffer.data);
    uint8_t* u8Data = static_cast<uint8_t*>(buffer.data);
    staticData = &u8Data[sizeof(MetaData)];
    itemData = &u8Data[sizeof(MetaData) + metaData->staticSize];

    return true;
}

/**
 * @brief delete all items for the buffer
 */
void
ItemBuffer::deleteAll()
{
    if(metaData == nullptr) {
        return;
    }

    while(m_lock.test_and_set(std::memory_order_acquire)) { asm(""); }
    for(uint64_t i = 0; i < metaData->itemCapacity; i++) {
        deleteItem(i);
    }
    m_lock.clear(std::memory_order_release);
}

/**
 * @brief initialize the item-list
 *
 * @param numberOfItems number of items to allocate
 * @param itemSize size of a single item
 *
 * @return false if values are invalid, else true
 */
bool
ItemBuffer::initDataBlocks(const uint64_t numberOfItems,
                           const uint32_t itemSize,
                           const uint64_t staticSize)
{
    // precheck
    if(itemSize == 0
            && staticSize == 0)
    {
        return false;
    }

    const uint64_t itemBytes = numberOfItems * itemSize;
    const uint64_t requiredBytes = itemBytes + staticSize + sizeof(MetaData);
    const uint64_t requiredNumberOfBlocks = calcBytesToBlocks(requiredBytes);

    // allocate blocks in buffer
    Hanami::allocateBlocks_DataBuffer(buffer, requiredNumberOfBlocks);
    buffer.usedBufferSize = requiredBytes;

    // init metadata object
    metaData = static_cast<MetaData*>(buffer.data);
    metaData[0] = MetaData();
    metaData->itemSize = itemSize;
    metaData->itemCapacity = numberOfItems;
    metaData->staticSize = staticSize;

    // init pointer
    uint8_t* u8Data = static_cast<uint8_t*>(buffer.data);
    staticData = &u8Data[sizeof(MetaData)];
    itemData = &u8Data[sizeof(MetaData) + staticSize];

    return true;
}

/**
 * @brief delete a specific item from the buffer by replacing it with a placeholder-item
 *
 * @param itemPos position of the item to delete
 *
 * @return false if buffer is invalid or position already deleted, else true
 */
bool
ItemBuffer::deleteItem(const uint64_t itemPos)
{
    // precheck
    if(metaData == nullptr
            || metaData->itemSize == 0
            || itemPos >= metaData->itemCapacity
            || metaData->numberOfItems == 0)
    {
        return false;
    }

    // get buffer
    uint8_t* blockBegin = static_cast<uint8_t*>(itemData);

    // data of the position
    const uint64_t currentBytePos = itemPos * metaData->itemSize;
    void* voidBuffer = static_cast<void*>(&blockBegin[currentBytePos]);
    EmptyPlaceHolder* placeHolder = static_cast<EmptyPlaceHolder*>(voidBuffer);

    // check that the position is active and not already deleted
    if(placeHolder->status == ItemBuffer::DELETED_SECTION) {
        return false;
    }

    // overwrite item with a placeholder and set the position as delted
    placeHolder->bytePositionOfNextEmptyBlock = ITEM_BUFFER_UNDEFINE_POS;
    placeHolder->status = ItemBuffer::DELETED_SECTION;

    // modify last place-holder
    const uint64_t blockPosition = metaData->bytePositionOfLastEmptyBlock;
    if(blockPosition != ITEM_BUFFER_UNDEFINE_POS)
    {
        voidBuffer = static_cast<void*>(&blockBegin[blockPosition]);
        EmptyPlaceHolder* lastPlaceHolder = static_cast<EmptyPlaceHolder*>(voidBuffer);
        lastPlaceHolder->bytePositionOfNextEmptyBlock = currentBytePos;
    }

    // set global values
    metaData->bytePositionOfLastEmptyBlock = currentBytePos;
    if(metaData->bytePositionOfFirstEmptyBlock == ITEM_BUFFER_UNDEFINE_POS) {
        metaData->bytePositionOfFirstEmptyBlock = currentBytePos;
    }

    metaData->numberOfItems--;

    return true;
}

/**
 * @brief try to reuse a deleted buffer segment
 *
 * @return item-position in the buffer, else UNINIT_STATE_32 if no empty space in buffer exist
 */
uint64_t
ItemBuffer::reuseItemPosition()
{
    // get byte-position of free space, if exist
    const uint64_t selectedPosition = metaData->bytePositionOfFirstEmptyBlock;
    if(selectedPosition == ITEM_BUFFER_UNDEFINE_POS) {
        return ITEM_BUFFER_UNDEFINE_POS;
    }

    // set pointer to the next empty space
    uint8_t* blockBegin = static_cast<uint8_t*>(itemData);
    void* voidBuffer = static_cast<void*>(&blockBegin[selectedPosition]);
    EmptyPlaceHolder* secetedPlaceHolder = static_cast<EmptyPlaceHolder*>(voidBuffer);
    metaData->bytePositionOfFirstEmptyBlock = secetedPlaceHolder->bytePositionOfNextEmptyBlock;

    // reset pointer, if no more free spaces exist
    if(metaData->bytePositionOfFirstEmptyBlock == ITEM_BUFFER_UNDEFINE_POS) {
        metaData->bytePositionOfLastEmptyBlock = ITEM_BUFFER_UNDEFINE_POS;
    }

    // convert byte-position to item-position and return this
    metaData->numberOfItems++;
    assert(selectedPosition % metaData->itemSize == 0);

    return selectedPosition / metaData->itemSize;
}

/**
 * @brief add a new forward-edge-section
 *
 * @return id of the new section, else UNINIT_STATE_32 if allocation failed
 */
uint64_t
ItemBuffer::reserveDynamicItem()
{
    // try to reuse item
    const uint64_t reusePos = reuseItemPosition();
    if(reusePos != ITEM_BUFFER_UNDEFINE_POS) {
        return reusePos;
    }

    // calculate size information
    const uint64_t numberOfBlocks = buffer.numberOfBlocks;
    const uint64_t itemBytes = (metaData->itemCapacity + 1) * metaData->itemSize;
    const uint64_t newNumberOfBlocks = calcBytesToBlocks(itemBytes);

    // allocate a new block, if necessary
    if(numberOfBlocks < newNumberOfBlocks) {
        Hanami::allocateBlocks_DataBuffer(buffer, newNumberOfBlocks - numberOfBlocks);
    }

    metaData->itemCapacity++;

    return metaData->itemCapacity-1;
}

}
