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

#include <api/endpoint_processing/items/value_item_map.h>

#include <hanami_common/items/table_item.h>

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
    for(const auto& [id, item] : other.m_valueMap) {
        m_valueMap.try_emplace(id, item);
    }

    // copy child-maps
    for(const auto& [id, itemMap] : other.m_childMaps)
    {
        ValueItemMap* newValue = new ValueItemMap(*itemMap);
        m_childMaps.try_emplace(id, newValue);
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
        for(const auto& [id, item] : other.m_valueMap) {;
            this->m_valueMap.try_emplace(id, item);
        }

        clearChildMap();

        // copy child-maps
        for(const auto& [id, itemMap] : other.m_childMaps) {
            this->m_childMaps.try_emplace(id, new ValueItemMap(*itemMap));
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
                     const json &value,
                     bool force)
{
    ValueItem valueItem;
    valueItem.item = value;
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
    const auto it = m_valueMap.find(key);
    if(it != m_valueMap.end()
            && force == false)
    {
        return false;
    }

    if(it != m_valueMap.end()) {
        it->second = value;
    } else {
        auto ret = m_valueMap.try_emplace(key, value);
        return ret.second;
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
    const auto it = m_childMaps.find(key);
    if(it != m_childMaps.end()
            && force == false)
    {
        return false;
    }

    if(it != m_childMaps.end()) {
        it->second = value;
    } else {
        auto ret = m_childMaps.try_emplace(key, value);
        return ret.second;
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
    if(m_valueMap.find(key) != m_valueMap.end()) {
        return true;
    }

    if(m_childMaps.find(key) != m_childMaps.end()) {
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
    const auto it = m_valueMap.find(key);
    if(m_valueMap.find(key) != m_valueMap.end())
    {
        m_valueMap.erase(it);
        return true;
    }

    const auto childIt = m_childMaps.find(key);
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
    const auto it = m_valueMap.find(key);
    if(it != m_valueMap.end()) {
        return it->second.item;
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
json
ValueItemMap::get(const std::string &key)
{
    const auto it = m_valueMap.find(key);
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
    auto it = m_valueMap.find(key);
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
    Hanami::TableItem table;
    table.addColumn("key");
    table.addColumn("value");

    // fill table
    for(const auto& [id, item] : m_valueMap) {
        table.addRow(std::vector<std::string>{id, item.item});
    }

    return table.toString();
}

/**
 * @brief ValueItemMap::clearChildMap
 */
void
ValueItemMap::clearChildMap()
{
    // clear old child map
    for(auto & [id, itemMap] : m_childMaps) {
        delete itemMap;
    }

    m_childMaps.clear();
}
