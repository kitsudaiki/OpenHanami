/**
 *  @file       item_buffer.h
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

#ifndef ITEM_BUFFER_H
#define ITEM_BUFFER_H

#include <hanami_common/buffer/data_buffer.h>

#include <vector>

namespace Hanami
{
constexpr uint64_t deleteMarker = 0x8000000000000000;
constexpr uint64_t undefinedPos = 0x7FFFFFFFFFFFFFFF;

template <typename T>
class ItemBuffer
{
   public:
    struct MetaData {
        uint32_t itemSize = 0;
        uint64_t itemCapacity = 0;
        uint64_t numberOfItems = 0;
        uint64_t firstEmptyBlock = undefinedPos;
        uint64_t lastEmptyBlock = undefinedPos;
    };

    T* items = nullptr;
    MetaData* metaData = nullptr;
    DataBuffer buffer = DataBuffer(1);

    ItemBuffer(){};
    ~ItemBuffer(){};

    /**
     * @brief initialize a new item-buffer based on the payload of an old one
     *
     * @param data pointer to the data to import
     * @param dataSize number of bytes of data
     *
     * @return true, if successful, else false if input is invalid
     */
    bool initBuffer(const void* data, const uint64_t dataSize)
    {
        // precheck
        if (dataSize == 0 || data == nullptr) {
            return false;
        }

        // allocate blocks in buffer and fill with old data
        Hanami::allocateBlocks_DataBuffer(buffer, calcBytesToBlocks(dataSize));
        buffer.usedBufferSize = dataSize;
        memcpy(buffer.data, data, dataSize);

        // init pointer
        metaData = static_cast<MetaData*>(buffer.data);
        uint8_t* u8Data = static_cast<uint8_t*>(buffer.data);
        items = reinterpret_cast<T*>(&u8Data[sizeof(MetaData)]);

        return true;
    }

    /**
     * @brief delete all items for the buffer
     */
    void deleteAll()
    {
        if (metaData == nullptr) {
            return;
        }

        for (uint64_t i = 0; i < metaData->itemCapacity; i++) {
            deleteItem(i);
        }
    }

    /**
     * @brief delete a specific item from the buffer by replacing it with a placeholder-item
     *
     * @param itemPos position of the item to delete
     *
     * @return false if buffer is invalid or position already deleted, else true
     */
    bool deleteItem(const uint64_t itemPos)
    {
        // precheck
        if (metaData == nullptr || metaData->itemSize == 0 || itemPos >= metaData->itemCapacity) {
            return false;
        }

        while (m_lock.test_and_set(std::memory_order_acquire)) {
            asm("");
        }

        if (metaData->numberOfItems == 0) {
            m_lock.clear(std::memory_order_release);
            return true;
        }

        // check if current position is already deleted
        const uint64_t link = m_links[itemPos];
        if ((link & deleteMarker) != 0) {
            m_lock.clear(std::memory_order_release);
            return false;
        }

        // delete current position
        m_links[itemPos] = deleteMarker | undefinedPos;
        items[itemPos] = T();

        // update the old last empty block
        const uint64_t lastEmptyBlock = metaData->lastEmptyBlock & undefinedPos;
        if (lastEmptyBlock != undefinedPos) {
            m_links[lastEmptyBlock] = deleteMarker | itemPos;
        }

        // update metadata
        metaData->lastEmptyBlock = itemPos;
        if (metaData->firstEmptyBlock == undefinedPos) {
            metaData->firstEmptyBlock = itemPos;
        }

        metaData->numberOfItems--;

        m_lock.clear(std::memory_order_release);

        return true;
    }

    /**
     * @brief initialize the item-list
     *
     * @param numberOfItems number of items to allocate
     * @param itemSize size of a single item
     *
     * @return false if values are invalid, else true
     */
    bool _initDataBlocks(const uint64_t numberOfItems, const uint32_t itemSize)
    {
        // precheck
        if (itemSize == 0) {
            return false;
        }

        const uint64_t itemBytes = numberOfItems * itemSize;
        const uint64_t requiredBytes = itemBytes + sizeof(MetaData);
        const uint64_t requiredNumberOfBlocks = calcBytesToBlocks(requiredBytes);

        // allocate blocks in buffer
        Hanami::allocateBlocks_DataBuffer(buffer, requiredNumberOfBlocks);
        buffer.usedBufferSize = requiredBytes;

        // init metadata object
        metaData = static_cast<MetaData*>(buffer.data);
        metaData[0] = MetaData();
        metaData->itemSize = itemSize;
        metaData->itemCapacity = numberOfItems;

        // init pointer
        uint8_t* u8Data = static_cast<uint8_t*>(buffer.data);
        items = reinterpret_cast<T*>(&u8Data[sizeof(MetaData)]);

        return true;
    }

    /**
     * @brief try to reuse a deleted buffer cluster
     *
     * @return item-position in the buffer, else UNINIT_STATE_32 if no empty space in buffer exist
     */
    uint64_t _reuseItemPosition()
    {
        // get byte-position of free space, if exist
        const uint64_t selectedPosition = metaData->firstEmptyBlock;
        if (selectedPosition == undefinedPos) {
            return undefinedPos;
        }

        // update metadata
        metaData->firstEmptyBlock = m_links[selectedPosition] & undefinedPos;
        m_links[selectedPosition] = undefinedPos;
        metaData->numberOfItems++;

        // reset pointer, if no more free spaces exist
        if (metaData->firstEmptyBlock == undefinedPos) {
            metaData->lastEmptyBlock = undefinedPos;
        }

        return selectedPosition;
    }

    /**
     * @brief try to reuse a deleted position in the buffer
     *
     * @return id of the new section, else 0x7FFFFFFFFFFFFFFF if allocation failed
     */
    uint64_t _reserveDynamicItem()
    {
        while (m_lock.test_and_set(std::memory_order_acquire)) {
            asm("");
        }

        // try to reuse item
        const uint64_t reusePos = _reuseItemPosition();
        if (reusePos != undefinedPos) {
            m_lock.clear(std::memory_order_release);
            return reusePos;
        }

        // calculate size information
        const uint64_t numberOfBlocks = buffer.numberOfBlocks;
        const uint64_t itemBytes = (metaData->itemCapacity + 1) * metaData->itemSize;
        const uint64_t newNumberOfBlocks = calcBytesToBlocks(itemBytes);

        // allocate a new block, if necessary
        if (numberOfBlocks < newNumberOfBlocks) {
            Hanami::allocateBlocks_DataBuffer(buffer, newNumberOfBlocks - numberOfBlocks);
        }

        metaData->itemCapacity++;
        const uint64_t ret = metaData->itemCapacity - 1;

        m_lock.clear(std::memory_order_release);

        return ret;
    }

    /**
     * @brief initialize buffer by allocating memory and init with default-items
     *
     * @param numberOfItems number of items to preallocate
     *
     * @return true, if successful, else false
     */
    bool initBuffer(const uint64_t numberOfItems)
    {
        // allocate memory
        const bool ret = _initDataBlocks(numberOfItems, sizeof(T));
        if (ret == false) {
            return false;
        }

        // set links to 0
        m_links.resize(numberOfItems, 0);
        std::fill(m_links.begin(), m_links.end(), undefinedPos);

        // init buffer with default-itemes
        const T newItem = T();
        std::fill_n(items, numberOfItems, newItem);

        // update metadata
        metaData->numberOfItems = numberOfItems;

        return true;
    }

    /**
     * @brief add a new items at an empty position inside of the buffer
     *
     * @param item new item to write into the buffer
     *
     * @return position inside of the buffer, where the new item was added, if successful, or
     *         2^64-1 if the buffer is already full
     */
    uint64_t addNewItem(const T& item)
    {
        // precheck
        if (metaData->itemSize == 0) {
            return undefinedPos;
        }

        // precheck
        if (metaData->numberOfItems >= metaData->itemCapacity) {
            return undefinedPos;
        }

        // get item-position inside of the buffer
        const uint64_t position = _reserveDynamicItem();
        if (position == undefinedPos) {
            return position;
        }

        // write new item at the position
        T* array = static_cast<T*>(items);
        array[position] = item;

        return position;
    }

   private:
    std::atomic_flag m_lock = ATOMIC_FLAG_INIT;
    std::vector<uint64_t> m_links;
};

/**
 * @brief get content of an item-buffer as array
 *
 * @param itembuffer pointer to items of buffer
 *
 * @return casted pointer of the item-buffer-content
 */
template <typename T>
inline T*
getItemData(ItemBuffer<T>& itembuffer)
{
    return static_cast<T*>(itembuffer.items);
}

}  // namespace Hanami

#endif  // ITEM_BUFFER_H
