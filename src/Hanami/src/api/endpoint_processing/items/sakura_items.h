/**
 * @file        sakura_items.h
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

#ifndef HANAMI_LANG_SAKURA_ITEMS_H
#define HANAMI_LANG_SAKURA_ITEMS_H

#include <vector>
#include <string>

#include <api/endpoint_processing/items/value_item_map.h>

namespace Hanami
{
class DataItem;
class DataMap;
struct DataBuffer;
}

//==================================================================================================
// SakuraItem
//==================================================================================================
class SakuraItem
{
public:
    enum ItemType
    {
        UNDEFINED_ITEM = 0,
        BLOSSOM_ITEM = 1,
        BLOSSOM_GROUP_ITEM = 2,
        TREE_ITEM = 4,
        SUBTREE_ITEM = 5,
        SEQUENTIELL_ITEM = 8,
        PARALLEL_ITEM = 9,
        IF_ITEM = 10,
        FOR_EACH_ITEM = 11,
        FOR_ITEM = 12,
        SEED_PART = 13
    };

    SakuraItem();
    virtual ~SakuraItem();
    virtual SakuraItem* copy() = 0;

    ItemType getType() const;
    ValueItemMap values;

protected:
    ItemType type = UNDEFINED_ITEM;
};

//==================================================================================================
// BlossomItem
//==================================================================================================
class BlossomItem : public SakuraItem
{
public:
    BlossomItem();
    ~BlossomItem();
    SakuraItem* copy();

    std::string blossomName = "";
    std::string blossomType = "";
    std::string blossomGroupType = "";
};

//==================================================================================================
// BlossomGroupItem
//==================================================================================================
class BlossomGroupItem : public SakuraItem
{
public:
    BlossomGroupItem();
    ~BlossomGroupItem();
    SakuraItem* copy();

    std::string id = "";
    std::string blossomGroupType = "";
    std::vector<std::string> nameHirarchie;

    std::vector<BlossomItem*> blossoms;
};

//==================================================================================================
// TreeItem
//==================================================================================================
class TreeItem : public SakuraItem
{
public:
    TreeItem();
    ~TreeItem();
    SakuraItem* copy();
    void getValidationMap(std::map<std::string, FieldDef> &validationMap) const;

    std::string id = "";

    std::string unparsedConent = "";
    std::string relativePath = "";
    std::string rootPath = "";
    std::string comment = "";
    std::vector<std::string> outputKeys;

    SakuraItem* childs;
};

//==================================================================================================
// SubtreeItem
//==================================================================================================
class SubtreeItem : public SakuraItem
{
public:
    SubtreeItem();
    ~SubtreeItem();
    SakuraItem* copy();

    std::string nameOrPath = "";
    Hanami::DataMap* parentValues = nullptr;

    // result
    std::vector<std::string> nameHirarchie;
};

//==================================================================================================
// IfBranching
//==================================================================================================
class IfBranching : public SakuraItem
{
public:
    enum compareTypes {
        EQUAL = 0,
        GREATER_EQUAL = 1,
        GREATER = 2,
        SMALLER_EQUAL = 3,
        SMALLER = 4,
        UNEQUAL = 5
    };

    IfBranching();
    ~IfBranching();
    SakuraItem* copy();

    ValueItem leftSide;
    compareTypes ifType = EQUAL;
    ValueItem rightSide;

    SakuraItem* ifContent = nullptr;
    SakuraItem* elseContent = nullptr;
};

//==================================================================================================
// ForEachBranching
//==================================================================================================
class ForEachBranching : public SakuraItem
{
public:
    ForEachBranching();
    ~ForEachBranching();
    SakuraItem* copy();

    std::string tempVarName = "";
    ValueItemMap iterateArray;
    bool parallel = false;

    SakuraItem* content = nullptr;
};

//==================================================================================================
// ForBranching
//==================================================================================================
class ForBranching : public SakuraItem
{
public:
    ForBranching();
    ~ForBranching();
    SakuraItem* copy();

    std::string tempVarName = "";
    ValueItem start;
    ValueItem end;
    bool parallel = false;

    SakuraItem* content = nullptr;
};

//==================================================================================================
// SequentiellPart
//==================================================================================================
class SequentiellPart : public SakuraItem
{
public:
    SequentiellPart();
    ~SequentiellPart();
    SakuraItem* copy();

    std::vector<SakuraItem*> childs;
};

//==================================================================================================
// ParallelPart
//==================================================================================================
class ParallelPart : public SakuraItem
{
public:
    ParallelPart();
    ~ParallelPart();
    SakuraItem* copy();

    SakuraItem* childs;
};

#endif // HANAMI_LANG_SAKURA_ITEMS_H
