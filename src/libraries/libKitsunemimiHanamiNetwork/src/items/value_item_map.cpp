/**
 * @file        value_item_map.h
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

#include <items/value_item_map.h>

#include <libKitsunemimiCommon/items/table_item.h>
#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi
{
namespace Hanami
{

/**
 * @brief constructor
 */
ValueItemMap::ValueItemMap() {}

/**
 * @brief destructor
 */
ValueItemMap::~ValueItemMap()
{
    clearChildMap();
}

/**
 * @brief copy-constructor
 */
ValueItemMap::ValueItemMap(const ValueItemMap &other)
{
    // copy items
    std::map<std::string, ValueItem>::const_iterator it;
    for(it = other.m_valueMap.begin();
        it != other.m_valueMap.end();
        it++)
    {
        ValueItem value = it->second;
        m_valueMap.insert(std::make_pair(it->first, value));
    }

    // copy child-maps
    std::map<std::string, ValueItemMap*>::const_iterator itChilds;
    for(itChilds = other.m_childMaps.begin();
        itChilds != other.m_childMaps.end();
        itChilds++)
    {
        ValueItemMap* newValue = new ValueItemMap(*itChilds->second);
        m_childMaps.insert(std::make_pair(itChilds->first, newValue));
    }
}

/**
 * @brief assignmet-operator
 */
ValueItemMap&
ValueItemMap::operator=(const ValueItemMap &other)
{
    if(this != &other)
    {
        // delet old items
        this->m_valueMap.clear();

        // copy items
        std::map<std::string, ValueItem>::const_iterator it;
        for(it = other.m_valueMap.begin();
            it != other.m_valueMap.end();
            it++)
        {
            ValueItem value = it->second;
            this->m_valueMap.insert(std::make_pair(it->first, value));
        }

        clearChildMap();

        // copy child-maps
        std::map<std::string, ValueItemMap*>::const_iterator itChilds;
        for(itChilds = other.m_childMaps.begin();
            itChilds != other.m_childMaps.end();
            itChilds++)
        {
            ValueItemMap* newValue = new ValueItemMap(*itChilds->second);
            this->m_childMaps.insert(std::make_pair(itChilds->first, newValue));
        }
    }

    return *this;
}

/**
 * @brief add a new key-value-pair to the map
 *
 * @param key key of the new entry
 * @param value data-item of the new entry
 * @param force true, to override, if key already exist.
 *
 * @return true, if new pair was inserted, false, if already exist and force-flag was false
 */
bool
ValueItemMap::insert(const std::string &key,
                     DataItem* value,
                     bool force)
{
    ValueItem valueItem;
    valueItem.item = value->copy();
    return insert(key, valueItem, force);
}

/**
 * @brief add a new key-value-pair to the map
 *
 * @param key key of the new entry
 * @param value value-item of the new entry
 * @param force true, to override, if key already exist.
 *
 * @return true, if new pair was inserted, false, if already exist and force-flag was false
 */
bool
ValueItemMap::insert(const std::string &key,
                     ValueItem &value,
                     bool force)
{
    std::map<std::string, ValueItem>::iterator it;
    it = m_valueMap.find(key);

    if(it != m_valueMap.end()
            && force == false)
    {
        return false;
    }

    if(it != m_valueMap.end()) {
        it->second = value;
    } else {
        m_valueMap.insert(std::make_pair(key, value));
    }

    return true;
}

/**
 * @brief add a new key-value-pair to the map
 *
 * @param key key of the new entry
 * @param value new child-map
 * @param force true, to override, if key already exist.
 *
 * @return true, if new pair was inserted, false, if already exist and force-flag was false
 */
bool
ValueItemMap::insert(const std::string &key,
                     ValueItemMap* value,
                     bool force)
{
    std::map<std::string, ValueItemMap*>::iterator it;
    it = m_childMaps.find(key);

    if(it != m_childMaps.end()
            && force == false)
    {
        return false;
    }

    if(it != m_childMaps.end()) {
        it->second = value;
    } else {
        m_childMaps.insert(std::make_pair(key, value));
    }

    return true;
}

/**
 * @brief check if the map contains a specific key
 *
 * @param key key to identify the entry
 *
 * @return true, if key exist inside the map, else false
 */
bool
ValueItemMap::contains(const std::string &key)
{
    std::map<std::string, ValueItem>::const_iterator it;
    it = m_valueMap.find(key);

    if(it != m_valueMap.end()) {
        return true;
    }

    std::map<std::string, ValueItemMap*>::const_iterator childIt;
    childIt = m_childMaps.find(key);

    if(childIt != m_childMaps.end()) {
        return true;
    }

    return false;
}

/**
 * @brief remove a value-item from the map
 *
 * @param key key to identify the entry
 *
 * @return true, if item was found and removed, else false
 */
bool
ValueItemMap::remove(const std::string &key)
{
    std::map<std::string, ValueItem>::const_iterator it;
    it = m_valueMap.find(key);

    if(it != m_valueMap.end())
    {
        m_valueMap.erase(it);
        return true;
    }

    std::map<std::string, ValueItemMap*>::const_iterator childIt;
    childIt = m_childMaps.find(key);

    if(childIt != m_childMaps.end())
    {
        m_childMaps.erase(childIt);
        return true;
    }

    return false;
}

/**
 * @brief get data-item inside a value-item of the map as string
 *
 * @param key key to identify the value
 *
 * @return item as string, if found, else empty string
 */
std::string
ValueItemMap::getValueAsString(const std::string &key)
{
    std::map<std::string, ValueItem>::const_iterator it;
    it = m_valueMap.find(key);
    if(it != m_valueMap.end()) {
        return it->second.item->toString();
    }

    return "";
}

/**
 * @brief get data-item inside a value-item of the map
 *
 * @param key key to identify the value
 *
 * @return pointer to the data-item, if found, else a nullptr
 */
DataItem*
ValueItemMap::get(const std::string &key)
{
    std::map<std::string, ValueItem>::const_iterator it;
    it = m_valueMap.find(key);
    if(it != m_valueMap.end()) {
        return it->second.item;
    }

    return nullptr;
}

/**
 * @brief get a value-item from the map
 *
 * @param key key to identify the value
 *
 * @return requested value-item, if found, else an empty uninitialized value-item
 */
ValueItem
ValueItemMap::getValueItem(const std::string &key)
{
    std::map<std::string, ValueItem>::const_iterator it;
    it = m_valueMap.find(key);
    if(it != m_valueMap.end()) {
        return it->second;
    }

    return ValueItem();
}

/**
 * @brief size get number of object in the map
 *
 * @return number of object inside the map
 */
uint64_t
ValueItemMap::size()
{
    return m_valueMap.size();
}

/**
 * @brief ValueItemMap::toString
 *
 * @return
 */
const std::string
ValueItemMap::toString()
{
    // init table output
    TableItem table;
    table.addColumn("key");
    table.addColumn("value");

    // fill table
    std::map<std::string, ValueItem>::const_iterator it;
    for(it = m_valueMap.begin();
        it != m_valueMap.end();
        it++)
    {
        table.addRow(std::vector<std::string>{it->first, it->second.item->toString()});
    }

    return table.toString();
}

/**
 * @brief ValueItemMap::getValidationMap
 * @param validationMap
 */
void
ValueItemMap::getValidationMap(std::map<std::string, FieldDef> &validationMap) const
{
    std::map<std::string, ValueItem>::const_iterator it;
    for(it = m_valueMap.begin();
        it != m_valueMap.end();
        it++)
    {
        ValueItem value = it->second;
        FieldDef::IO_ValueType ioType = FieldDef::INPUT_TYPE;
        if(value.type == ValueItem::OUTPUT_PAIR_TYPE) {
            ioType = FieldDef::OUTPUT_TYPE;
        }
        const bool isReq = value.item->getString() == "?";

        validationMap.emplace(it->first, FieldDef(ioType, value.fieldType, isReq, value.comment));
    }
}

/**
 * @brief ValueItemMap::clearChildMap
 */
void
ValueItemMap::clearChildMap()
{
    // clear old child map
    std::map<std::string, ValueItemMap*>::const_iterator itChilds;
    for(itChilds = m_childMaps.begin();
        itChilds != m_childMaps.end();
        itChilds++)
    {
        ValueItemMap* oldMap = itChilds->second;
        delete oldMap;
    }

    m_childMaps.clear();
}

} // namespace Hanami
} // namespace Kitsunemimi
