/**
 *  @file       obj_creator.cpp
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  MIT License
 */

#include "obj_creator.h"

namespace Hanami
{

/**
 * @brief constructor
 */
ObjCreator::ObjCreator() {}

/**
 * @brief converts an obj-item into a string
 *
 * @param convertedString reference to the resulting string
 * @param input obj-item, which should be converted
 *
 * @return resulting string
 */
bool
ObjCreator::create(std::string &convertedString,
                   const ObjItem &input)
{
    convertedString = "";

    // convert vertizes
    for(uint32_t i = 0; i < input.vertizes.size(); i++)
    {
        convertedString += "v ";
        convertedString += std::to_string(input.vertizes.at(i).x) + " ";
        convertedString += std::to_string(input.vertizes.at(i).y) + " ";
        convertedString += std::to_string(input.vertizes.at(i).z) + "\n";
    }

    // convert textures
    for(uint32_t i = 0; i < input.textures.size(); i++)
    {
        convertedString += "vt ";
        convertedString += std::to_string(input.textures.at(i).x) + " ";
        convertedString += std::to_string(input.textures.at(i).y) + "\n";
    }

    // convert normals
    for(uint32_t i = 0; i < input.normals.size(); i++)
    {
        convertedString += "vn ";
        convertedString += std::to_string(input.normals.at(i).x) + " ";
        convertedString += std::to_string(input.normals.at(i).y) + " ";
        convertedString += std::to_string(input.normals.at(i).z) + "\n";
    }

    // convert points
    for(uint32_t i = 0; i < input.points.size(); i++)
    {
        convertedString += "p " + std::to_string(input.points.at(i)) + "\n";
    }

    // convert lines
    for(uint32_t i = 0; i < input.lines.size(); i++)
    {
        convertedString += "l";
        for(uint32_t j = 0; j < input.lines.at(i).size(); j++)
        {
            const uint32_t id = input.lines.at(i).at(j);
            convertedString += " " + std::to_string(id);
        }
        convertedString += "\n";
    }

    // convert faces
    for(uint32_t i = 0; i < input.faces.size(); i++)
    {
        convertedString += "f";
        for(uint32_t j = 0; j < input.faces.at(i).size(); j++)
        {
            // v
            const int32_t v = input.faces.at(i).at(j).v;
            convertedString += " " + std::to_string(v) + "/";

            // vt
            const int32_t vt = input.faces.at(i).at(j).vt;
            if(vt > 0) {
                convertedString += std::to_string(vt);
            }

            // vn
            const int32_t vn = input.faces.at(i).at(j).vn;
            if(vn > 0)
            {
                convertedString += "/";
                convertedString += std::to_string(vn);
            }
        }
        convertedString += "\n";
    }

    return true;
}

}
