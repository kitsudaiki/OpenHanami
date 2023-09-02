/**
 *  @file       obj_item.h
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  MIT License
 */

#include <hanami_obj/obj_item.h>

#include <obj_parser.h>
#include <obj_creator.h>

namespace Hanami
{

/**
 * @brief parse an obj-string
 *
 * @param result empty obj-item for the parsed information
 * @param input input-string, which should be parsed
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
parseString(ObjItem &result,
            const std::string &input,
            ErrorContainer &error)
{
    ObjParser parser;
    return parser.parse(result, input, error);
}

/**
 * @brief converts an obj-item into a string
 *
 * @param convertedString reference to the resulting string
 * @param input obj-item, which should be converted
 *
 * @return resulting string
 */

bool
convertToString(std::string &convertedString,
                const ObjItem &input)
{
    ObjCreator creator;
    return creator.create(convertedString, input);
}

}
