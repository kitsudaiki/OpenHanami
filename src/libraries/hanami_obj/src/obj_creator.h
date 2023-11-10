/**
 *  @file       obj_creator.h
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  MIT License
 */

#ifndef OBJ_CREATOR_H
#define OBJ_CREATOR_H

#include <hanami_obj/obj_item.h>

namespace Hanami
{

class ObjCreator
{
   public:
    ObjCreator();

    bool create(std::string& convertedString, const ObjItem& input);
};

}  // namespace Hanami

#endif  // OBJ_CREATOR_H
