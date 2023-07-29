/**
 *  @file    json_item.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <libKitsunemimiJson/json_item.h>

#include <libKitsunemimiCommon/items/data_items.h>
#include <json_parsing/json_parser_interface.h>

using Kitsunemimi::DataItem;
using Kitsunemimi::DataArray;
using Kitsunemimi::DataValue;
using Kitsunemimi::DataMap;

namespace Kitsunemimi
{

/**
 * @brief JsonItem::JsonItem
 */
JsonItem::JsonItem()
{
    m_content = nullptr;
    m_deletable = true;
}

/**
 * @brief creates a new item based on an already existring item
 *
 * @param otherItem other item, which have to be copied
 */
JsonItem::JsonItem(const JsonItem &otherItem)
{
    if(otherItem.m_content != nullptr) {
        m_content = otherItem.m_content->copy();
    } else {
        m_content = nullptr;
    }
}

/**
 * @brief creates a new item
 *
 * @param dataItem pointer to an already existring abstract-json
 */
JsonItem::JsonItem(DataItem* dataItem,
                   const bool copy)
{
    if(copy)
    {
        if(dataItem != nullptr) {
            m_content = dataItem->copy();
        }
    }
    else
    {
        m_content = dataItem;
        m_deletable = false;
    }
}

/**
 * @brief creates an object-item
 *
 * @param value map for the new object, which can be empty
 */
JsonItem::JsonItem(std::map<std::string, JsonItem> &value)
{
    m_content = new DataMap();
    for(const auto& [id, item] : value) {
        insert(id, item);
    }
}

/**
 * @brief creates an array-item
 *
 * @param value vector for the new object, which can be empty
 */
JsonItem::JsonItem(std::vector<JsonItem> &value)
{
    m_content = new DataArray();
    for(const JsonItem &item : value) {
        append(item);
    }
}

JsonItem::JsonItem(const char* value)
{
    m_content = new DataValue(value);
}

JsonItem::JsonItem(const std::string &value)
{
    m_content = new DataValue(value);
}

JsonItem::JsonItem(const int value)
{
    m_content = new DataValue(value);
}

JsonItem::JsonItem(const float value)
{
    m_content = new DataValue(value);
}

JsonItem::JsonItem(const long value)
{
    m_content = new DataValue(value);
}

JsonItem::JsonItem(const double value)
{
    m_content = new DataValue(value);
}

JsonItem::JsonItem(const bool value)
{
    m_content = new DataValue(value);
}

/**
 * destructor
 */
JsonItem::~JsonItem()
{
    clear();
}

/**
 * @brief convert a json-formated string into a json-object-tree
 *
 * @param input json-formated string, which should be parsed
 * @param error reference for error-message output
 *
 * @return true, if successful, else false
 */
bool
JsonItem::parse(const std::string &input,
                ErrorContainer &error)
{
    JsonParserInterface* parser = JsonParserInterface::getInstance();

    // parse ini-template into a json-tree
    DataItem* result = nullptr;
    if(input.size() > 0) {
        result = parser->parse(input, error);
    } else {
        result = new DataMap();
    }

    // process a failure
    if(result == nullptr) {
        return false;
    }

    clear();

    m_content = result;

    return true;
}

/**
 * @brief replace the content of the item with the content of another item
 *
 * @param other other item, which have to be copied
 *
 * @return pointer to this current item
 */
JsonItem&
JsonItem::operator=(const JsonItem &other)
{
    if(this != &other)
    {
        clear();
        if(other.m_content != nullptr) {
            m_content = other.m_content->copy();
        } else {
            m_content = other.m_content;
        }
    }

    return *this;
}


/**
 * @brief replace the content of the item with the content of another item
 *
 * @param other other item, which have to be copied
 *
 * @return pointer to this current item
 */
JsonItem&
JsonItem::operator=(const DataItem* other)
{
    clear();
    if(other != nullptr) {
        m_content = other->copy();
    } else {
        m_content = nullptr;
    }

    return *this;
}

/**
 * @brief writes a new string-value into the json-value
 */
bool
JsonItem::setValue(const char* value)
{
    if(m_content == nullptr) {
        m_content = new DataValue();
    }

    if(m_content->getType() == DataItem::VALUE_TYPE)
    {
        m_content->toValue()->setValue(value);
        return true;
    }

    return false;
}

/**
 * @brief writes a new string-value into the json-value
 */
bool
JsonItem::setValue(const std::string &value)
{
    if(m_content == nullptr) {
        m_content = new DataValue();
    }

    if(m_content->getType() == DataItem::VALUE_TYPE)
    {
        m_content->toValue()->setValue(value);
        return true;
    }

    return false;
}

/**
 * @brief writes a new integer-value into the json-value
 */
bool
JsonItem::setValue(const int &value)
{
    if(m_content == nullptr) {
        m_content = new DataValue();
    }

    if(m_content->getType() == DataItem::VALUE_TYPE)
    {
        m_content->toValue()->setValue(value);
        return true;
    }

    return false;
}

/**
 * @brief writes a new float-value into the json-value
 */
bool
JsonItem::setValue(const float &value)
{
    if(m_content == nullptr) {
        m_content = new DataValue();
    }

    if(m_content->getType() == DataItem::VALUE_TYPE)
    {
        m_content->toValue()->setValue(value);
        return true;
    }

    return false;
}

/**
 * @brief writes a new long-value into the json-value
 */
bool
JsonItem::setValue(const long &value)
{
    if(m_content == nullptr) {
        m_content = new DataValue();
    }

    if(m_content->getType() == DataItem::VALUE_TYPE)
    {
        m_content->toValue()->setValue(value);
        return true;
    }

    return false;
}

/**
 * @brief writes a new double-value into the json-value
 */
bool
JsonItem::setValue(const double &value)
{
    if(m_content == nullptr) {
        m_content = new DataValue();
    }

    if(m_content->getType() == DataItem::VALUE_TYPE)
    {
        m_content->toValue()->setValue(value);
        return true;
    }

    return false;
}

/**
 * @brief writes a new bool-value into the json-value
 */
bool
JsonItem::setValue(const bool &value)
{
    if(m_content == nullptr) {
        m_content = new DataValue();
    }

    if(m_content->getType() == DataItem::VALUE_TYPE)
    {
        m_content->toValue()->setValue(value);
        return true;
    }

    return false;
}

/**
 * @brief insert a key-value-pair if the current item is a json-object
 *
 * @param key key of the new pair
 * @param value new json-item-object
 * @param force true to overwrite an existing key (default: false)
 *
 * @return false, if item is not a json-object, else true
 */
bool
JsonItem::insert(const std::string &key,
                 const JsonItem &value,
                 const bool force)
{
    if(value.m_content == nullptr
            || key == "")
    {
        return false;
    }

    if(m_content == nullptr) {
        m_content = new DataMap();
    }

    if(m_content->getType() == DataItem::MAP_TYPE)
    {
        return m_content->toMap()->insert(key,
                                          value.m_content->copy(),
                                          force);
    }

    return false;
}

/**
 * @brief add a new item, if the current item is a json-array
 *
 * @param value new json-item-object
 *
 * @return false, if item is not a json-array, else true
 */
bool
JsonItem::append(const JsonItem &value)
{
    if(value.m_content == nullptr) {
        return false;
    }

    if(m_content == nullptr) {
        m_content = new DataArray();
    }

    if(m_content->getType() == DataItem::ARRAY_TYPE)
    {
        m_content->toArray()->append(value.m_content->copy());
        return true;
    }

    return false;
}

/**
 * @brief replace an item within a array
 *
 * @param index position in the array
 * @param value new json-item-object
 *
 * @return false, if items not valid of index too high, else true
 */
bool
JsonItem::replaceItem(const uint32_t index,
                      const JsonItem &value)
{
    if(value.m_content == nullptr) {
        return false;
    }

    if(m_content == nullptr) {
        m_content = new DataArray();
    }

    if(m_content->getType() == DataItem::ARRAY_TYPE
            && m_content->toArray()->array.size() > index)
    {
        delete m_content->toArray()->array[index];
        m_content->toArray()->array[index] = value.m_content->copy();
        return true;
    }

    return false;
}

/**
 * @brief JsonItem::deleteContent
 *
 * @return
 */
bool
JsonItem::deleteContent()
{
    if(m_content == nullptr) {
        return false;
    }

    delete m_content;
    m_content = nullptr;

    return true;
}

/**
 * @brief JsonItem::getItemContent
 * @return
 */
DataItem*
JsonItem::getItemContent() const
{
    return m_content;
}

/**
 * @brief steal the content of the item to avoid copy the content, when the json-item is not longer
 *        necessary
 *
 * @return content of the json-item
 */
DataItem*
JsonItem::stealItemContent()
{
    DataItem* tempVar = m_content;
    m_content = nullptr;
    return tempVar;
}

/**
 * @brief get a specific entry of the item
 *
 * @param key key of the requested value
 *
 * @return nullptr if index in key is to high, else object
 */
JsonItem
JsonItem::operator[](const std::string key)
{
    if(m_content == nullptr) {
        return JsonItem();
    }

    return JsonItem(m_content->get(key));
}

/**
 * @brief get a specific item of the object
 *
 * @param index index of the item
 *
 * @return nullptr if index is to high, else object
 */
JsonItem
JsonItem::operator[](const uint32_t index)
{
    if(m_content == nullptr) {
        return JsonItem();
    }

    return JsonItem(m_content->get(index));
}

/**
 * @brief get a specific entry of the item
 *
 * @param key key of the requested value
 *
 * @return nullptr if index in key is to high, else object
 */
JsonItem
JsonItem::get(const std::string key,
              const bool copy) const
{
    if(m_content == nullptr) {
        return JsonItem();
    }

    return JsonItem(m_content->get(key), copy);
}

/**
 * @brief get a specific item of the object
 *
 * @param index index of the item
 *
 * @return nullptr if index is to high, else object
 */
JsonItem
JsonItem::get(const uint32_t index,
              const bool copy) const
{
    if(m_content == nullptr) {
        return JsonItem();
    }

    return JsonItem(m_content->get(index), copy);
}

/**
 * @brief get string of the item
 *
 * @return string, of the item if int-type, else empty string
 */
const std::string
JsonItem::getString() const
{
    if(m_content == nullptr) {
        return "";
    }

    if(m_content->getType() == DataItem::VALUE_TYPE) {
        return m_content->toValue()->getString();
    }

    return "";
}

/**
 * @brief get int-value of the item
 *
 * @return int-value, of the item if int-type, else 0
 */
int
JsonItem::getInt() const
{
    if(m_content == nullptr) {
        return 0;
    }

    if(m_content->toValue()->getValueType() == DataItem::INT_TYPE) {
        return m_content->toValue()->getInt();
    }

    return 0;
}

/**
 * @brief get float-value of the item
 *
 * @return float-value, of the item if float-type, else 0
 */
float
JsonItem::getFloat() const
{
    if(m_content == nullptr) {
        return 0.0f;
    }

    if(m_content->toValue()->getValueType() == DataItem::FLOAT_TYPE) {
        return m_content->toValue()->getFloat();
    }

    return 0.0f;
}

/**
 * @brief get long-value of the item
 *
 * @return long-value, of the item if int-type, else 0
 */
long
JsonItem::getLong() const
{
    if(m_content == nullptr) {
        return 0;
    }

    if(m_content->toValue()->getValueType() == DataItem::INT_TYPE) {
        return m_content->toValue()->getLong();
    }

    return 0;
}

/**
 * @brief get double-value of the item
 *
 * @return double-value, of the item if float-type, else 0
 */
double
JsonItem::getDouble() const
{
    if(m_content == nullptr) {
        return 0.0;
    }

    if(m_content->toValue()->getValueType() == DataItem::FLOAT_TYPE) {
        return m_content->toValue()->getDouble();
    }

    return 0.0;
}

/**
 * @brief get bool-value of the item
 *
 * @return bool-value, of the item if bool-type, else false
 */
bool
JsonItem::getBool() const
{
    if(m_content == nullptr) {
        return false;
    }

    if(m_content->toValue()->getValueType() == DataItem::BOOL_TYPE) {
        return m_content->toValue()->getBool();
    }

    return false;
}

/**
 * @brief getter for the number of elements in the item
 *
 * @return number of elements in the item
 */
uint64_t
JsonItem::size() const
{
    if(m_content == nullptr) {
        return 0;
    }

    return m_content->size();
}

/**
 * @brief get list of keys if the json-item is an json-object
 *
 * @return string-list with the keys of the map
 */
const std::vector<std::string>
JsonItem::getKeys() const
{
    if(m_content == nullptr) {
        return std::vector<std::string>();
    }

    if(m_content->getType() == DataItem::MAP_TYPE) {
        return m_content->toMap()->getKeys();
    }

    return std::vector<std::string>();
}

/**
 * @brief check if a key is in the object-map
 *
 * @param key key-string which should be searched in the map of the object-item
 *
 * @return false if the key doesn't exist or the item is no json-object, else true
 */
bool
JsonItem::contains(const std::string &key) const
{
    if(m_content == nullptr) {
        return false;
    }

    if(m_content->getType() == DataItem::MAP_TYPE) {
        return m_content->toMap()->contains(key);
    }

    return false;
}

/**
 * @brief check if the current item is valid
 *
 * @return false, if the item is a null-pointer, else true
 */
bool
JsonItem::isValid() const
{
    if(m_content != nullptr) {
        return true;
    }
    return false;
}

/**
 * @brief check if the current item is null
 *
 * @return true, if the item is a null-value, else false
 */
bool
JsonItem::isNull() const
{
    if(m_content == nullptr) {
        return true;
    }
    return false;
}

/**
 * @brief check if current item is a object
 *
 * @return true if current item is a json-object, else false
 */
bool
JsonItem::isMap() const
{
    if(m_content == nullptr) {
        return false;
    }

    return m_content->isMap();
}

/**
 * @brief check if current item is a array
 *
 * @return true if current item is a json-array, else false
 */
bool JsonItem::isArray() const
{
    if(m_content == nullptr) {
        return false;
    }

    return m_content->isArray();
}

/**
 * @brief check if current item is a value
 *
 * @return true if current item is a json-value, else false
 */
bool JsonItem::isValue() const
{
    if(m_content == nullptr) {
        return false;
    }

    return m_content->isValue();
}

/**
 * @brief check if current item is a string-value
 *
 * @return true if current item is a string-value, else false
 */
bool
JsonItem::isString() const
{
    if(m_content == nullptr) {
        return false;
    }

    if(m_content->getType() == DataItem::VALUE_TYPE)  {
        return m_content->toValue()->isStringValue();
    }

    return false;
}

/**
 * @brief check if current item is a float-value
 *
 * @return true if current item is a float-value, else false
 */
bool
JsonItem::isFloat() const
{
    if(m_content == nullptr) {
        return false;
    }

    if(m_content->getType() == DataItem::VALUE_TYPE)  {
        return m_content->toValue()->isFloatValue();
    }

    return false;
}

/**
 * @brief check if current item is a int-value
 *
 * @return true if current item is a int-value, else false
 */
bool
JsonItem::isInteger() const
{
    if(m_content == nullptr) {
        return false;
    }

    if(m_content->getType() == DataItem::VALUE_TYPE)  {
        return m_content->toValue()->isIntValue();
    }

    return false;
}

/**
 * @brief check if current item is a bool-value
 *
 * @return true if current item is a bool-value, else false
 */
bool
JsonItem::isBool() const
{
    if(m_content == nullptr) {
        return false;
    }

    if(m_content->getType() == DataItem::VALUE_TYPE)  {
        return m_content->toValue()->isBoolValue();
    }

    return false;
}

/**
 * @brief remove an item from the key-value-list
 *
 * @param key key of the pair, which should be deleted
 *
 * @return false if the key doesn't exist, else true
 */
bool
JsonItem::remove(const std::string &key)
{
    if(m_content != nullptr) {
        return m_content->remove(key);
    }

    return false;
}

/**
 * @brief remove an item from the object
 *
 * @param index index of the item
 *
 * @return false if index is to high, else true
 */
bool
JsonItem::remove(const uint32_t index)
{
    if(m_content != nullptr) {
        return m_content->remove(index);
    }
    return false;
}

/**
 * @brief prints the content of the object
 *
 * @param output pointer to the output-string on which the object-content as string will be appended
 */
const std::string
JsonItem::toString(bool indent) const
{
    if(m_content != nullptr) {
        return m_content->toString(indent);
    }
    return "";
}

/**
 * @brief delete the underlaying json-object
 */
void
JsonItem::clear()
{
    if(m_content != nullptr
            && m_deletable)
    {
        delete m_content;
        m_content = nullptr;
    }
}

}
