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

#define ITEM_BUFFER_UNDEFINE_POS 0xFFFFFFFFFFFFFFFF

namespace Hanami
{

class ItemBuffer;
template <typename T>
inline T* getItemData(ItemBuffer& itembuffer);

class ItemBuffer
{
   public:
    enum SectionStatus {
        UNDEFINED_SECTION = 0,
        ACTIVE_SECTION = 1,
        DELETED_SECTION = 2,
    };

    struct MetaData {
        uint32_t itemSize = 0;
        uint64_t itemCapacity = 0;
        uint64_t staticSize = 0;
        uint64_t numberOfItems = 0;
        uint64_t bytePositionOfFirstEmptyBlock = ITEM_BUFFER_UNDEFINE_POS;
        uint64_t bytePositionOfLastEmptyBlock = ITEM_BUFFER_UNDEFINE_POS;
    };

    void* staticData = nullptr;
    void* itemData = nullptr;
    MetaData* metaData = nullptr;
    DataBuffer buffer = DataBuffer(1);

    ItemBuffer();
    ~ItemBuffer();

    /**
     * @brief initialize buffer by allocating memory and init with default-items
     *
     * @param numberOfItems number of items to preallocate
     *
     * @return true, if successful, else false
     */
    template <typename T>
    bool initBuffer(const uint64_t numberOfItems, const uint64_t staticSize = 0)
    {
        // allocate memory
        const bool ret = initDataBlocks(numberOfItems, sizeof(T), staticSize);
        if (ret == false) {
            return false;
        }

        // init buffer with default-itemes
        T* items = static_cast<T*>(itemData);
        T newItem = T();
        std::fill_n(items, numberOfItems, newItem);
        metaData->numberOfItems = numberOfItems;

        return true;
    }

    bool initBuffer(const uint64_t staticSize);
    bool initBuffer(const void* data, const uint64_t dataSize);

    /**
     * @brief add a new items at an empty position inside of the buffer
     *
     * @param item new item to write into the buffer
     *
     * @return position inside of the buffer, where the new item was added, if successful, or
     *         2^64-1 if the buffer is already full
     */
    template <typename T>
    uint64_t addNewItem(const T& item)
    {
        // precheck
        if (metaData->itemSize == 0) {
            return ITEM_BUFFER_UNDEFINE_POS;
        }

        // init invalid default-value
        uint64_t position = ITEM_BUFFER_UNDEFINE_POS;

        // precheck
        if (metaData->numberOfItems >= metaData->itemCapacity) {
            return position;
        }

        while (m_lock.test_and_set(std::memory_order_acquire)) {
            asm("");
        }

        // get item-position inside of the buffer
        position = reserveDynamicItem();
        if (position == ITEM_BUFFER_UNDEFINE_POS) {
            return position;
        }

        // write new item at the position
        T* array = static_cast<T*>(itemData);
        array[position] = item;

        m_lock.clear(std::memory_order_release);

        return position;
    }

    bool deleteItem(const uint64_t itemPos);
    void deleteAll();

   private:
    std::atomic_flag m_lock = ATOMIC_FLAG_INIT;

    bool initDataBlocks(const uint64_t numberOfItems,
                        const uint32_t itemSize,
                        const uint64_t staticSize);

    uint64_t reuseItemPosition();
    uint64_t reserveDynamicItem();
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
getItemData(ItemBuffer& itembuffer)
{
    return static_cast<T*>(itembuffer.itemData);
}

}  // namespace Hanami

#endif  // ITEM_BUFFER_H
