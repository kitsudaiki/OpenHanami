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

#ifndef HANAMI_LANG_VALUE_ITEM_MAP_H
#define HANAMI_LANG_VALUE_ITEM_MAP_H

#include <api/endpoint_processing/blossom.h>
#include <api/endpoint_processing/items/value_items.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

class ValueItemMap
{
   public:
    ValueItemMap();
    ~ValueItemMap();
    ValueItemMap(const ValueItemMap &other);
    ValueItemMap &operator=(const ValueItemMap &other);

    // add and remove
    bool insert(const std::string &key, const json &value, bool force = true);
    bool insert(const std::string &key, ValueItem &value, bool force = true);
    bool insert(const std::string &key, ValueItemMap *value, bool force = true);
    bool remove(const std::string &key);

    // getter
    bool contains(const std::string &key);
    std::string getValueAsString(const std::string &key);
    json get(const std::string &key);
    ValueItem getValueItem(const std::string &key);
    uint64_t size();
    const std::string toString();

    // internal value-maps
    std::map<std::string, ValueItem> m_valueMap;
    std::map<std::string, ValueItemMap *> m_childMaps;

   private:
    void clearChildMap();
};

#endif  // HANAMI_LANG_VALUE_ITEM_MAP_H
