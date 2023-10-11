/**
 *  @file       obj_parser.h
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  MIT License
 */

#include "obj_parser.h"

#include <hanami_common/logger.h>
#include <hanami_common/methods/string_methods.h>
#include <hanami_common/methods/vector_methods.h>

namespace Hanami
{

/**
 * @brief default-constructor
 */
ObjParser::ObjParser() {}

/**
 * @brief parse an obj-formated string
 *
 * @param result empty obj-item for the parsed information
 * @param input input-string, which should be parsed
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ObjParser::parse(ObjItem &result, const std::string &inputString, ErrorContainer &error)
{
    // copy and prepare input string
    std::string preparedString = inputString;
    Hanami::replaceSubstring(preparedString, "\t", " ");

    // split string into the single lines
    std::vector<std::string> splittedContent;
    Hanami::splitStringByDelimiter(splittedContent, preparedString, '\n');

    // iterate of the lines of the input
    for (uint64_t i = 0; i < splittedContent.size(); i++) {
        // skip empty lines
        if (splittedContent.at(i).size() == 0) {
            continue;
        }

        // split line into the single parts
        std::vector<std::string> splittedLine;
        Hanami::splitStringByDelimiter(splittedLine, splittedContent.at(i), ' ');
        Hanami::removeEmptyStrings(splittedLine);

        bool state = false;

        // handle vertex
        if (splittedLine.at(0) == "v") {
            Vec4 vertex;
            state = parseVertex(vertex, splittedLine);
            result.vertizes.push_back(vertex);
        }

        // handle textures
        if (splittedLine.at(0) == "vt") {
            Vec4 texture;
            state = parseVertex(texture, splittedLine);
            result.textures.push_back(texture);
        }

        // handle normals
        if (splittedLine.at(0) == "vn") {
            Vec4 normale;
            state = parseVertex(normale, splittedLine);
            result.normals.push_back(normale);
        }

        // handle point
        if (splittedLine.at(0) == "p") {
            int32_t value = 0;
            state = parseInt(value, splittedLine.at(1));
            state = state && value > 0;
            result.points.push_back(static_cast<uint32_t>(value));
        }

        // handle line
        if (splittedLine.at(0) == "l") {
            std::vector<uint32_t> indizes;
            state = parseValueList(indizes, splittedLine);
            result.lines.push_back(indizes);
        }

        // handle face
        if (splittedLine.at(0) == "f") {
            std::vector<Index> indizes;
            state = parseIndexList(indizes, splittedLine);
            result.faces.push_back(indizes);
        }

        // check result
        if (state == false) {
            error.addMeesage("ERROR while parsing obj-file content in line " + std::to_string(i));
            LOG_ERROR(error);
            return false;
        }
    }

    return true;
}

/**
 * @brief parse coordinate
 *
 * @param result reference to the vec4-variable, where the converted value should be written into
 * @param lineContent splitted content of the line
 *
 * @return true, if successful, else false
 */
bool
ObjParser::parseVertex(Vec4 &result, const std::vector<std::string> &lineContent)
{
    // precheck
    if (lineContent.size() < 3) {
        return false;
    }

    bool ret = true;

    // parse coordinates
    ret = ret && parseFloat(result.x, lineContent.at(1));
    ret = ret && parseFloat(result.y, lineContent.at(2));
    if (lineContent.size() > 3) {
        ret = ret && parseFloat(result.z, lineContent.at(3));
    }

    return ret;
}

/**
 * @brief parse list of values
 *
 * @param result reference to the value-list, where the converted value should be written into
 * @param lineContent splitted content of the line
 *
 * @return true, if successful, else false
 */
bool
ObjParser::parseValueList(std::vector<uint32_t> &result,
                          const std::vector<std::string> &lineContent)
{
    // precheck
    if (lineContent.size() < 4) {
        return false;
    }

    // iterate over the line
    for (uint32_t i = 1; i < lineContent.size(); i++) {
        // converts the parts into an index-item
        int32_t newIndex;
        bool ret = parseInt(newIndex, lineContent.at(i));
        if (ret == false) {
            return false;
        }

        result.push_back(static_cast<uint32_t>(newIndex));
    }

    return true;
}

/**
 * @brief parse list of indizes
 *
 * @param result reference to the index-list, where the converted value should be written into
 * @param lineContent splitted content of the line
 *
 * @return true, if successful, else false
 */
bool
ObjParser::parseIndexList(std::vector<Index> &result, const std::vector<std::string> &lineContent)
{
    // precheck
    if (lineContent.size() < 4) {
        return false;
    }

    // iterate over the line
    for (uint32_t i = 1; i < lineContent.size(); i++) {
        // split index-entry into its parts
        std::vector<std::string> indexList;
        Hanami::splitStringByDelimiter(indexList, lineContent.at(i), '/');

        // converts the parts into an index-item
        Index newIndex;
        bool ret = parseIndex(newIndex, indexList);
        if (ret == false) {
            return false;
        }

        result.push_back(newIndex);
    }

    return true;
}

/**
 * @brief converts the parts into an index-item
 *
 * @param result reference to the index-variable, where the converted value should be written into
 * @param indexContent string-list with the single parts of the index
 *
 * @return true, if successful, else false
 */
bool
ObjParser::parseIndex(Index &result, const std::vector<std::string> &indexContent)
{
    bool ret = true;

    // convert v
    if (indexContent.size() > 0) {
        if (indexContent.at(0).size() > 0) {
            ret = ret && parseInt(result.v, indexContent.at(0));
        }
    }

    // convert vt
    if (indexContent.size() > 1) {
        if (indexContent.at(1).size() > 0) {
            ret = ret && parseInt(result.vt, indexContent.at(1));
        }
    }

    // convert vn
    if (indexContent.size() > 2) {
        if (indexContent.at(2).size() > 0) {
            ret = ret && parseInt(result.vn, indexContent.at(2));
        }
    }

    return ret;
}

/**
 * @brief convert a string into a float-value
 *
 * @param result reference to the float-variable, where the converted value should be written into
 * @param input input-string to parse
 *
 * @return true, if successful, else false
 */
bool
ObjParser::parseFloat(float &result, const std::string &input)
{
    char *err = nullptr;
    result = std::strtof(input.c_str(), &err);
    return std::string(err).size() == 0;
}

/**
 * @brief convert a string into an int-value
 *
 * @param result reference to the int-variable, where the converted value should be written into
 * @param input input-string to parse
 *
 * @return true, if successful, else false
 */
bool
ObjParser::parseInt(int &result, const std::string &input)
{
    char *err = nullptr;
    result = static_cast<int32_t>(std::strtol(input.c_str(), &err, 10));
    return std::string(err).size() == 0;
}

}  // namespace Hanami
