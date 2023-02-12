/**
 *  @file       data_items.cpp
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

#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi
{

//==================================================================================================
// AbstractData
//==================================================================================================

DataItem::~DataItem() {}

/**
 * @brief request type, of the current data-object
 *
 * @return object-specific entry of the dataTypes-enumeration
 */
DataItem::dataTypes
DataItem::getType() const
{
    return m_type;
}

/**
 * @brief check if DataItem is a DataValue
 */
bool
DataItem::isValue() const
{
    if(m_type == VALUE_TYPE) {
        return true;
    }

    return false;
}

/**
 * @brief check if DataItem is a DataMap
 */
bool
DataItem::isMap() const
{
    if(m_type == MAP_TYPE) {
        return true;
    }

    return false;
}

/**
 * @brief check if DataItem is a DataArray
 */
bool
DataItem::isArray() const
{
    if(m_type == ARRAY_TYPE) {
        return true;
    }

    return false;
}

/**
 * @brief check if DataItem is a String-Value
 */
bool
DataItem::isStringValue() const
{
    if(m_valueType == STRING_TYPE) {
        return true;
    }

    return false;
}

/**
 * @brief check if DataItem is a int-Value
 */
bool
DataItem::isIntValue() const
{
    if(m_valueType == INT_TYPE) {
        return true;
    }

    return false;
}

/**
 * @brief check if DataItem is a float-Value
 */
bool
DataItem::isFloatValue() const
{
    if(m_valueType == FLOAT_TYPE) {
        return true;
    }

    return false;
}

/**
 * @brief check if DataItem is a bool-Value
 */
bool
DataItem::isBoolValue() const
{
    if(m_valueType == BOOL_TYPE) {
        return true;
    }

    return false;
}

/**
 * @brief convert to a DataArray
 */
DataArray*
DataItem::toArray()
{
    if(m_type == ARRAY_TYPE) {
        return static_cast<DataArray*>(this);
    }

    return nullptr;
}

/**
 * @brief convert to a DataMap
 */
DataMap*
DataItem::toMap()
{
    if(m_type == MAP_TYPE) {
        return static_cast<DataMap*>(this);
    }

    return nullptr;
}

/**
 * @brief convert to a DataVolue
 */
DataValue*
DataItem::toValue()
{
    if(m_type == VALUE_TYPE) {
        return static_cast<DataValue*>(this);
    }

    return nullptr;
}

/**
 * @brief request the string of the data-value, if it is from string-type
 *
 * @return string of the data-value, if data-value is from string-type, else empty string
 */
const std::string
DataItem::getString() const
{
    if(m_valueType == STRING_TYPE)
    {
        const DataValue* value = dynamic_cast<const DataValue*>(this);
        return std::string(value->content.stringValue);
    }

    return std::string("");
}

/**
 * @brief request the integer of the data-value, if it is from int-type
 *
 * @return integer of the data-value, if data-value is from int-type, else empty 0
 */
int
DataItem::getInt()
{
    if(m_valueType == INT_TYPE)
    {
        DataValue* value = dynamic_cast<DataValue*>(this);
        return static_cast<int>(value->content.longValue);
    }

    return 0;
}

/**
 * @brief request the flaot of the data-value, if it is from float-type
 *
 * @return float of the data-value, if data-value is from float-type, else empty 0.0
 */
float
DataItem::getFloat()
{
    if(m_valueType == FLOAT_TYPE)
    {
        DataValue* value = dynamic_cast<DataValue*>(this);
        return static_cast<float>(value->content.doubleValue);
    }

    return 0.0f;
}

/**
 * @brief request the integer of the data-value, if it is from int-type
 *
 * @return integer of the data-value, if data-value is from int-type, else empty 0
 */
long
DataItem::getLong()
{
    if(m_valueType == INT_TYPE)
    {
        DataValue* value = dynamic_cast<DataValue*>(this);
        return value->content.longValue;
    }

    return 0l;
}

/**
 * @brief request the flaot of the data-value, if it is from float-type
 *
 * @return float of the data-value, if data-value is from float-type, else empty 0.0
 */
double
DataItem::getDouble()
{
    if(m_valueType == FLOAT_TYPE)
    {
        DataValue* value = dynamic_cast<DataValue*>(this);
        return value->content.doubleValue;
    }

    return 0.0;
}

/**
 * @brief request the bool of the data-value, if it is from bool-type
 *
 * @return bool of the data-value, if data-value is from bool-type, else empty flase
 */
bool
DataItem::getBool()
{
    if(m_valueType == BOOL_TYPE)
    {
        DataValue* value = dynamic_cast<DataValue*>(this);
        return value->content.boolValue;
    }

    return false;
}

/**
 * @brief add indent and linebreak to be better human-readable
 */
void
DataItem::addIndent(std::string* output,
                    const bool indent,
                    const uint32_t level) const
{
    if(indent == true)
    {
        output->append("\n");
        for(uint32_t i = 0; i < level; i++) {
            output->append("    ");
        }
    }
}

//==================================================================================================
// DataValue
//==================================================================================================

/**
 * @brief DataValue::DataValue
 */
DataValue::DataValue()
{
    m_type = VALUE_TYPE;
    m_valueType = STRING_TYPE;

    content.stringValue = new char[1];
    content.stringValue[0] = '\0';
}

/**
 * @brief data-value for char-arrays
 */
DataValue::DataValue(const char* text)
{
    m_type = VALUE_TYPE;
    m_valueType = STRING_TYPE;

    if(text == nullptr)
    {
        content.stringValue = new char[1];
        content.stringValue[0] = '\0';
    }
    else
    {
        const size_t len = strlen(text);

        content.stringValue = new char[len + 1];
        memcpy(content.stringValue, text, len);
        content.stringValue[len] = '\0';
    }
}

/**
 * @brief data-value for strings
 */
DataValue::DataValue(const std::string &text)
{
    m_type = VALUE_TYPE;
    m_valueType = STRING_TYPE;

    content.stringValue = new char[text.size() + 1];
    memcpy(content.stringValue, text.c_str(), text.size());
    content.stringValue[text.size()] = '\0';
}

/**
 * @brief data-value for integers
 */
DataValue::DataValue(const int value)
{
    m_type = VALUE_TYPE;
    m_valueType = INT_TYPE;
    content.longValue = value;
}

/**
 * @brief data-value for float
 */
DataValue::DataValue(const float value)
{
    m_type = VALUE_TYPE;
    m_valueType = FLOAT_TYPE;
    content.doubleValue = static_cast<double>(value);
}

/**
 * @brief data-value for long
 */
DataValue::DataValue(const long value)
{
    m_type = VALUE_TYPE;
    m_valueType = INT_TYPE;
    content.longValue = value;
}

/**
 * @brief data-value for double
 */
DataValue::DataValue(const double value)
{
    m_type = VALUE_TYPE;
    m_valueType = FLOAT_TYPE;
    content.doubleValue = value;
}

/**
 * @brief data-value for bool
 */
DataValue::DataValue(const bool value)
{
    m_type = VALUE_TYPE;
    m_valueType = BOOL_TYPE;
    content.boolValue = value;
}

/**
 * @brief copy-assingment-constructor
 */
DataValue::DataValue(const DataValue &other)
{
    clearDataValue();

    // copy meta-data
    m_type = other.m_type;
    m_valueType = other.m_valueType;

    // copy content
    if(other.m_valueType == STRING_TYPE)
    {
        const size_t len = strlen(other.content.stringValue);

        content.stringValue = new char[len + 1];
        strncpy(content.stringValue, other.content.stringValue, len);
        content.stringValue[len] = '\0';

        assert(strlen(other.content.stringValue) == strlen(content.stringValue));
    }
    else
    {
        content = other.content;
    }
}

/**
 * @brief destructor
 */
DataValue::~DataValue()
{
    clearDataValue();
}

/**
 * @brief assignment operator
 */
DataValue
&DataValue::operator=(const DataValue &other)
{
    if(this != &other)
    {
        clearDataValue();

        // copy meta-data
        this->m_type = other.m_type;
        this->m_valueType = other.m_valueType;

        // copy content
        if(other.m_valueType == STRING_TYPE)
        {
            const size_t len = strlen(other.content.stringValue);

            this->content.stringValue = new char[len + 1];
            strncpy(this->content.stringValue, other.content.stringValue, len);
            this->content.stringValue[len] = '\0';

            assert(strlen(other.content.stringValue) == strlen(this->content.stringValue));
        }
        else
        {
            this->content = other.content;
        }
    }

    return *this;
}

/**
 * @brief get type inside the data-value
 *
 * @return value-type
 */
DataValue::dataValueTypes
DataValue::getValueType()
{
    return m_valueType;
}

/**
 * @brief fake-method which exist here only for the inheritance and returns everytime nullptr
 */
DataItem*
DataValue::operator[](const std::string &) const
{
    return nullptr;
}

/**
 * @brief fake-method which exist here only for the inheritance and returns everytime nullptr
 */
DataItem*
DataValue::operator[](const uint64_t) const
{
    return nullptr;
}

/**
 * @brief fake-method which exist here only for the inheritance and returns everytime nullptr
 */
DataItem*
DataValue::get(const std::string &) const
{
    return nullptr;
}

/**
 * @brief fake-method which exist here only for the inheritance and returns everytime nullptr
 */
DataItem*
DataValue::get(const uint64_t) const
{
    return nullptr;
}

/**
 * @brief return size of the values
 *
 * @return length of the string, if string-typed value, else 0
 */
uint64_t
DataValue::size() const
{
    if(m_valueType == STRING_TYPE) {
        return getString().size();
    }
    return 0;
}

/**
 * @brief fake-method which exist here only for the inheritance and returns everytime false
 */
bool
DataValue::remove(const std::string&)
{
    return false;
}

/**
 * @brief fake-method which exist here only for the inheritance and returns everytime false
 */
bool
DataValue::remove(const uint64_t)
{
    return false;
}

/**
 * @brief reset content to int-type with value 0
 */
void
DataValue::clear()
{
    clearDataValue();
}

/**
 * @brief reset content to int-type with value 0
 */
void
DataValue::clearDataValue()
{
    if(m_valueType == STRING_TYPE
            && content.stringValue != nullptr)
    {
        delete[] content.stringValue;
    }

    m_type = VALUE_TYPE;
    m_valueType = INT_TYPE;
    content.longValue = 0l;
}

/**
 * @brief copy the data-value
 *
 * @return pointer to a copy of the value
 */
DataItem*
DataValue::copy() const
{
    DataValue* tempItem = nullptr;

    if(m_valueType == STRING_TYPE) {
        tempItem = new DataValue(std::string(content.stringValue));
    }

    if(m_valueType == INT_TYPE) {
        tempItem = new DataValue(content.longValue);
    }

    if(m_valueType == FLOAT_TYPE) {
        tempItem = new DataValue(content.doubleValue);
    }

    if(m_valueType == BOOL_TYPE) {
        tempItem = new DataValue(content.boolValue);
    }

    return tempItem;
}

/**
 * @brief return the content as string
 */
const std::string
DataValue::toString(const bool,
                    std::string* output,
                    const uint32_t) const
{
    std::string out = "";
    if(output == nullptr) {
        output = &out;
    }

    if(m_valueType == STRING_TYPE)
    {
        output->append(std::string(content.stringValue));
    }

    if(m_valueType == INT_TYPE) {
        output->append(std::to_string(content.longValue));
    }

    if(m_valueType == FLOAT_TYPE) {
        output->append(std::to_string(content.doubleValue));
    }

    if(m_valueType == BOOL_TYPE)
    {
        if(content.boolValue) {
            output->append("true");
        } else {
            output->append("false");
        }
    }

    return out;
}

/**
 * @brief writes a new string into the data-value
 */
void
DataValue::setValue(const char* value)
{
    if(m_valueType == STRING_TYPE) {
        delete[] content.stringValue;
    }

    m_type = VALUE_TYPE;
    m_valueType = STRING_TYPE;

    if(value == nullptr)
    {
        content.stringValue = new char[1];
        content.stringValue[0] = '\0';
    }
    else
    {
        const size_t len = strlen(value);

        content.stringValue = new char[len + 1];
        memcpy(content.stringValue, value, len);
        content.stringValue[len] = '\0';
    }
}

/**
 * @brief writes a new string into the data-value
 */
void
DataValue::setValue(const std::string &value)
{
    if(m_valueType == STRING_TYPE) {
        delete[] content.stringValue;
    }

    m_type = VALUE_TYPE;
    m_valueType = STRING_TYPE;

    content.stringValue = new char[value.size() + 1];
    memcpy(content.stringValue, value.c_str(), value.size());
    content.stringValue[value.size()] = '\0';
}

/**
 * @brief writes a new integer into the data-value
 */
void
DataValue::setValue(const int &value)
{
    if(m_valueType == STRING_TYPE) {
        delete[] content.stringValue;
    }

    m_type = VALUE_TYPE;
    m_valueType = INT_TYPE;

    content.longValue = value;
}

/**
 * @brief writes a new float into the data-value
 */
void
DataValue::setValue(const float &value)
{
    if(m_valueType == STRING_TYPE) {
        delete content.stringValue;
    }

    m_type = VALUE_TYPE;
    m_valueType = FLOAT_TYPE;

    content.doubleValue = static_cast<double>(value);
}

/**
 * @brief writes a new long integer into the data-value
 */
void
DataValue::setValue(const long &value)
{
    if(m_valueType == STRING_TYPE) {
        delete content.stringValue;
    }

    m_type = VALUE_TYPE;
    m_valueType = INT_TYPE;

    content.longValue = value;
}

/**
 * @brief writes a new double into the data-value
 */
void
DataValue::setValue(const double &value)
{
    if(m_valueType == STRING_TYPE) {
        delete[] content.stringValue;
    }

    m_type = VALUE_TYPE;
    m_valueType = FLOAT_TYPE;

    content.doubleValue = value;
}

/**
 * @brief writes a new boolean into the data-value
 */
void
DataValue::setValue(const bool &value)
{
    if(m_valueType == STRING_TYPE) {
        delete content.stringValue;
    }

    m_type = VALUE_TYPE;
    m_valueType = BOOL_TYPE;

    content.boolValue = value;
}

//==================================================================================================
// DataMap
//==================================================================================================

/**
 * @brief object for key-value-pairs
 */
DataMap::DataMap()
{
    m_type = MAP_TYPE;
}

/**
 * @brief copy-assingment-constructor
 */
DataMap::DataMap(const DataMap &other)
{
    std::map<std::string, DataItem*> otherMap = other.map;

    // clear old map
    clearDataMap();

    // copy meta-data
    m_type = other.m_type;
    m_valueType = other.m_valueType;

    // copy content
    std::map<std::string, DataItem*>::iterator it;
    for(it = otherMap.begin();
        it != otherMap.end();
        it++)
    {
        if(it->second != nullptr) {
            map.insert(std::make_pair(it->first, it->second->copy()));
        } else {
            map.insert(std::make_pair(it->first, nullptr));
        }
    }
}

/**
 * @brief delete all items in the key-value-list
 */
DataMap::~DataMap()
{
    clearDataMap();
}

/**
 * @brief assignment operator
 */
DataMap
&DataMap::operator=(const DataMap &other)
{
    if(this != &other)
    {
        std::map<std::string, DataItem*> otherMap = other.map;

        // clear old map
        clearDataMap();

        // copy meta-data
        this->m_type = other.m_type;
        this->m_valueType = other.m_valueType;

        // copy content
        for(auto const& [name, item] : otherMap)
        {
            if(item != nullptr) {
                this->map.insert(make_pair(name, item->copy()));
            } else {
                this->map.insert(std::make_pair(name, nullptr));
            }
        }
    }

    return *this;
}

/**
 * @brief get a specific item of the object
 *
 * @return nullptr if index in key is to high, else object
 */
DataItem*
DataMap::operator[](const std::string &key) const
{
    return get(key);
}

/**
 * @brief get a specific item of the object
 *
 * @return nullptr if index is to high, else object
 */
DataItem*
DataMap::operator[](const uint64_t index) const
{
    return get(index);
}

/**
 * @brief get a specific item of the object
 *
 * @return nullptr if index in key is to high, else object
 */
DataItem*
DataMap::get(const std::string &key) const
{
    std::map<std::string, DataItem*>::const_iterator it;
    it = map.find(key);

    if(it != map.end()) {
        return it->second;
    }

    return nullptr;
}

/**
 * @brief get a specific item of the object
 *
 * @return nullptr if index is to high, else object
 */
DataItem*
DataMap::get(const uint64_t index) const
{
    if(map.size() <= index) {
        return nullptr;
    }

    uint32_t counter = 0;
    std::map<std::string, DataItem*>::const_iterator it;
    for(it = map.begin();
        it != map.end();
        it++)
    {
        if(counter == index) {
            return it->second;
        }
        counter++;
    }

    return nullptr;
}

/**
 * @brief getter for the number of elements in the key-value-list
 *
 * @return number of elements in the key-value-list
 */
uint64_t
DataMap::size() const
{
    return map.size();
}

/**
 * @brief get list of keys of the objects-map
 *
 * @return string-list with the keys of the map
 */
const std::vector<std::string>
DataMap::getKeys() const
{
    std::vector<std::string> result;
    std::map<std::string, DataItem*>::const_iterator it;
    for(it = map.begin();
        it != map.end();
        it++)
    {
        result.push_back(it->first);
    }

    return result;
}

/**
 * @brief get list of values of the objects-map
 *
 * @return DataItem-list with the keys of the map
 */
const std::vector<DataItem*>
DataMap::getValues() const
{
    std::vector<DataItem*> result;
    std::map<std::string, DataItem*>::const_iterator it;
    for(it = map.begin();
        it != map.end();
        it++)
    {
        result.push_back(it->second);
    }

    return result;
}

/**
 * @brief check if a key is in the object-map
 *
 * @return false if the key doesn't exist, else true
 */
bool
DataMap::contains(const std::string &key) const
{
    std::map<std::string, DataItem*>::const_iterator it;
    it = map.find(key);
    if(it != map.end()) {
        return true;
    }

    return false;
}

/**
 * @brief get the string-value behind the key inside the data-map
 */
const std::string
DataMap::getStringByKey(const std::string &key) const
{
    DataItem* item = get(key);
    if(item == nullptr) {
        return std::string("");
    }

    return item->getString();
}

/**
 * @brief get the bool-value behind the key inside the data-map
 */
bool
DataMap::getBoolByKey(const std::string &key) const
{
    DataItem* item = get(key);
    if(item == nullptr) {
        return false;
    }

    return item->getBool();
}

/**
 * @brief get the int-value behind the key inside the data-map
 */
int
DataMap::getIntByKey(const std::string &key) const
{
    DataItem* item = get(key);
    if(item == nullptr) {
        return 0;
    }

    return item->getInt();
}

/**
 * @brief get the float-value behind the key inside the data-map
 */
float
DataMap::getFloatByKey(const std::string &key) const
{
    DataItem* item = get(key);
    if(item == nullptr) {
        return 0.0f;
    }

    return item->getFloat();
}

/**
 * @brief get the long-value behind the key inside the data-map
 */
long
DataMap::getLongByKey(const std::string &key) const
{
    DataItem* item = get(key);
    if(item == nullptr) {
        return 0l;
    }

    return item->getLong();
}

/**
 * @brief get the double-value behind the key inside the data-map
 */
double
DataMap::getDoubleByKey(const std::string &key) const
{
    DataItem* item = get(key);
    if(item == nullptr) {
        return 0.0;
    }

    return item->getDouble();
}

/**
 * @brief remove an item from the key-value-list
 *
 * @return false if the key doesn't exist, else true
 */
bool
DataMap::remove(const std::string &key)
{
    std::map<std::string, DataItem*>::const_iterator it;
    it = map.find(key);

    if(it != map.end())
    {
        if(it->second != nullptr) {
            delete it->second;
        }
        map.erase(it);
        return true;
    }

    return false;
}

/**
 * @brief remove an item from the object
 *
 * @return false if index is to high, else true
 */
bool
DataMap::remove(const uint64_t index)
{
    if(map.size() <= index) {
        return false;
    }

    uint32_t counter = 0;
    std::map<std::string, DataItem*>::const_iterator it;
    for(it = map.begin();
        it != map.end();
        it++)
    {
        if(counter == index)
        {
            if(it->second != nullptr) {
                delete it->second;
            }
            map.erase(it);
            return true;
        }
        counter++;
    }

    return false;
}

/**
 * @brief delete all elements from the map
 */
void
DataMap::clear()
{
    clearDataMap();
}

/**
 * @brief delete all elements from the map
 */
void
DataMap::clearDataMap()
{
    std::map<std::string, DataItem*>::iterator it;
    for(it = map.begin();
        it != map.end();
        it++)
    {
        DataItem* tempItem = it->second;
        if(tempItem != nullptr) {
            delete tempItem;
        }
    }

    map.clear();
}

/**
 * @brief copy the object with all elements
 *
 * @return pointer to a copy of the object
 */
DataItem*
DataMap::copy() const
{
    DataMap* tempItem = new DataMap();
    std::map<std::string, DataItem*>::const_iterator it;
    for(it = map.begin();
        it != map.end();
        it++)
    {
        if(it->second == nullptr) {
            tempItem->insert(it->first, nullptr);
        } else {
            tempItem->insert(it->first, it->second->copy());
        }
    }

    return tempItem;
}

/**
 * @brief return the content as string
 */
const std::string
DataMap::toString(const bool indent,
                  std::string* output,
                  const uint32_t level) const
{
    std::string out = "";

    if(output == nullptr) {
        output = &out;
    }

    // begin map
    bool firstRun = false;
    output->append("{");

    std::map<std::string, DataItem*>::const_iterator it;
    for(it = map.begin();
        it != map.end();
        it++)
    {
        if(firstRun) {
            output->append(",");
        }
        firstRun = true;

        // add key
        addIndent(output, indent, level + 1);
        output->append("\"");
        output->append(it->first);
        output->append("\"");

        output->append(":");

        if(indent == true) {
            output->append(" ");
        }

        // add value
        if(it->second == nullptr)
        {
            output->append("null");
        }
        else
        {
            // if value is string-item, then set quotes
            if(it->second->isStringValue()) {
                output->append("\"");
            }

            // convert value of item into stirng
            it->second->toString(indent, output, level + 1);

            // if value is string-item, then set quotes
            if(it->second->isStringValue()) {
                output->append("\"");
            }
        }
    }

    // close map
    addIndent(output, indent, level);
    output->append("}");

    return out;
}

/**
 * @brief add new key-value-pair to the object
 *
 * @return false if key already exist, else true
 */
bool
DataMap::insert(const std::string &key,
                DataItem* value,
                bool force)
{
    // check if key already exist
    std::map<std::string, DataItem*>::iterator it;
    it = map.find(key);
    if(it != map.end()
            && force == false)
    {
        return false;
    }

    // if already exist and should be overwritten,
    // then delete the old one at first and insert the new one
    if(it != map.end())
    {
        if(it->second != nullptr) {
            delete it->second;
        }
        it->second = value;
    }
    else
    {
        map.insert(std::make_pair(key, value));
    }

    return true;
}

//==================================================================================================
// DataArray
//==================================================================================================

/**
 * @brief array for items in data-style
 */
DataArray::DataArray()
{
    m_type = ARRAY_TYPE;
    array.reserve(5);
}

/**
 * @brief copy-assingment-constructor
 */
DataArray::DataArray(const DataArray &other)
{
    array.reserve(5);

    // clear old array
    clearDataArray();

    // copy meta-data
    m_type = other.m_type;
    m_valueType = other.m_valueType;

    // copy content
    for(uint32_t i = 0; i < other.array.size(); i++)
    {
        if(other.array[i] != nullptr) {
            array.push_back(other.array[i]->copy());
        } else {
            array.push_back(nullptr);
        }
    }
}

/**
 * @brief delete all items in the array
 */
DataArray::~DataArray()
{
    clearDataArray();
}

/**
 * @brief copy-assignment operator
 */
DataArray
&DataArray::operator=(const DataArray &other)
{
    if(this != &other)
    {
        // clear old array
        clearDataArray();

        // copy meta-data
        this->m_type = other.m_type;
        this->m_valueType = other.m_valueType;

        // copy content
        for(uint32_t i = 0; i < other.array.size(); i++)
        {
            if(other.array[i] != nullptr) {
                this->array.push_back(other.array[i]->copy());
            } else {
                this->array.push_back(nullptr);
            }
        }
    }

    return *this;
}

/**
 * @brief get a specific item of the array
 *
 * @return nullptr if index in key is to high, else true
 */
DataItem*
DataArray::operator[](const std::string &key) const
{
    return get(key);
}

/**
 * @brief get a specific item of the array
 *
 * @return nullptr if index is to high, else true
 */
DataItem*
DataArray::operator[](const uint64_t index) const
{
    return get(index);
}

/**
 * @brief get a specific item of the array
 *
 * @return nullptr if index in key is to high, else object
 */
DataItem*
DataArray::get(const std::string &) const
{
    return nullptr;
}

/**
 * @brief get a specific item of the array
 *
 * @return nullptr if index is to high, else the object
 */
DataItem*
DataArray::get(const uint64_t index) const
{
    if(array.size() <= index) {
        return nullptr;
    }

    return array[index];
}

/**
 * @brief getter for the number of elements in the array
 *
 * @return number of elements in the array
 */
uint64_t
DataArray::size() const
{
    return array.size();
}

/**
 * @brief remove an item from the array
 *
 * @return false if index in key is to high, else true
 */
bool
DataArray::remove(const std::string &key)
{
    const uint32_t index = static_cast<uint32_t>(std::stoi(key));

    if(array.size() <= index) {
        return false;
    }

    DataItem* tempItem = array[index];
    if(tempItem != nullptr) {
        delete tempItem;
    }

    array.erase(array.begin() + index);

    return true;
}

/**
 * @brief remove an item from the array
 *
 * @return false if index is to high, else true
 */
bool
DataArray::remove(const uint64_t index)
{
    if(array.size() <= index) {
        return false;
    }

    DataItem* tempItem = array[index];
    if(tempItem != nullptr) {
        delete tempItem;
    }

    array.erase(array.begin() + static_cast<uint32_t>(index));

    return true;
}

/**
 * @brief delete all elements from the array
 */
void
DataArray::clear()
{
    clearDataArray();
}

/**
 * @brief DataArray::clearDataArray
 */
void
DataArray::clearDataArray()
{
    for(uint32_t i = 0; i < this->array.size(); i++)
    {
        DataItem* tempItem = this->array[i];
        if(tempItem != nullptr) {
            delete tempItem;
        }
    }

    this->array.clear();
}

/**
 * @brief copy the array with all elements
 *
 * @return pointer to a copy of the array
 */
DataItem*
DataArray::copy() const
{
    DataArray* tempItem = new DataArray();

    for(uint32_t i = 0; i < array.size(); i++)
    {
        if(array[i] == nullptr) {
            tempItem->append(nullptr);
        } else {
            tempItem->append(array[i]->copy());
        }
    }

    return tempItem;
}

/**
 * @brief return the content as string
 */
const std::string
DataArray::toString(const bool indent,
                    std::string* output,
                    const uint32_t level) const
{
    std::string out = "";

    if(output == nullptr) {
        output = &out;
    }

    // begin array
    output->append("[");
    addIndent(output, indent, level + 1);

    std::vector<DataItem*>::const_iterator it;
    for(it = array.begin(); it != array.end(); it++)
    {
        // separate items of the array with comma
        if(it != array.begin())
        {
            output->append(",");
            addIndent(output, indent, level + 1);
        }

        // add value
        if((*it) == nullptr)
        {
            output->append("null");
        }
        else
        {
            // if value is string-item, then set quotes
            if((*it)->isStringValue()) {
                output->append("\"");
            }

            // convert value of item into stirng
            (*it)->toString(indent, output, level + 1);

            // if value is string-item, then set quotes
            if((*it)->isStringValue()) {
                output->append("\"");
            }
        }
    }

    // close array
    addIndent(output, indent, level);
    output->append("]");

    return out;
}

/**
 * @brief add a new item to the array
 */
void
DataArray::append(DataItem* item)
{
    array.push_back(item);
}

}  // namespace Kitsunemimi
