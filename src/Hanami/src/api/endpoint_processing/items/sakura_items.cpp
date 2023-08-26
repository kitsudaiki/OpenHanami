/**
 * @file        sakura_items.cpp
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

#include <api/endpoint_processing/items/sakura_items.h>

//===================================================================
// SakuraItem
//===================================================================
SakuraItem::SakuraItem() {}

SakuraItem::~SakuraItem() {}

SakuraItem::ItemType SakuraItem::getType() const
{
    return type;
}

//===================================================================
// BlossomItem
//===================================================================
BlossomItem::BlossomItem()
{
    type = BLOSSOM_ITEM;
}

BlossomItem::~BlossomItem() {}

SakuraItem*
BlossomItem::copy()
{
    BlossomItem* newItem = new BlossomItem();

    newItem->type = type;
    newItem->values = values;

    newItem->blossomName = blossomName;
    newItem->blossomGroupType = blossomGroupType;
    newItem->blossomType = blossomType;

    return newItem;
}
