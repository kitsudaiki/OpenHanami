/**
 *  @file       obj_item.cpp
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  MIT License
 */

#ifndef OBJ_ITEM_H
#define OBJ_ITEM_H

#include <vector>
#include <string>

#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{

struct Vec4
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct Index
{
    int32_t v = -1;
    int32_t vt = -1;
    int32_t vn = -1;
};

struct ObjItem
{
    std::vector<Vec4> vertizes;
    std::vector<Vec4> normals;
    std::vector<Vec4> textures;
    std::vector<uint32_t> points;
    std::vector<std::vector<uint32_t>> lines;
    std::vector<std::vector<Index>> faces;

    ObjItem() {}
};

bool parseString(ObjItem &result,
                 const std::string &input,
                 ErrorContainer &error);
bool convertToString(std::string &convertedString, const ObjItem &input);

}

#endif // OBJ_ITEM_H
