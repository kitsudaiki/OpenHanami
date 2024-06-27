/**
 *  @file       obj_parser.cpp
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  MIT License
 */

#ifndef OBJ_PARSER_H
#define OBJ_PARSER_H

#include <hanami_common/logger.h>
#include <hanami_obj/obj_item.h>

namespace Hanami
{

class ObjParser
{
   public:
    ObjParser();

    bool parse(ObjItem& result, const std::string& inputString, ErrorContainer& error);

   private:
    bool parseIndizes(ObjItem& result, const std::string& inputString);

    bool parseVertex(Vec4& result, const std::vector<std::string>& lineContent);
    bool parseValueList(std::vector<uint32_t>& result, const std::vector<std::string>& lineContent);
    bool parseIndexList(std::vector<Index>& result, const std::vector<std::string>& lineContent);
    bool parseIndex(Index& result, const std::vector<std::string>& indexContent);

    bool parseFloat(float& result, const std::string& input);
    bool parseInt(int& result, const std::string& input);
};

}  // namespace Hanami

#endif  // OBJ_PARSER_H
