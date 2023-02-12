/**
 * @file       segment_meta.cpp
 *
 * @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright  Apache License Version 2.0
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

#include <libKitsunemimiHanamiSegmentParser/segment_meta.h>
#include <segment_parsing/segment_parser_interface.h>

namespace Kitsunemimi::Hanami
{

/**
 * @brief parse a segment-template string
 *
 * @param result pointer to the resulting object, which should be filled
 * @param input segment-template string, which should be parsed
 * @param error reference for error-message output
 *
 * @return true, if successful, else false
 */
bool
parseSegment(SegmentMeta* result,
             const std::string &input,
             ErrorContainer &error)
{
    SegmentParserInterface* parser = SegmentParserInterface::getInstance();

    if(input.size() == 0)
    {
        error.addMeesage("Parsing of segment-template failed, because the input is empty");
        return false;
    }

    return parser->parse(result, input, error);
}

}
