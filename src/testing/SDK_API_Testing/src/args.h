/**
 * @file        args.h
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

#ifndef TSUGUMITESTER_ARGS_H
#define TSUGUMITESTER_ARGS_H

#include <hanami_args/arg_parser.h>
#include <hanami_common/logger.h>

/**
 * @brief register cli-arguments
 *
 * @param argparser reference to argument parser
 *
 * @return true if successful, else false
 */
bool
registerArguments(Hanami::ArgParser* argparser)
{
    std::string helpText = "";

    // config-flag
    argparser->registerString("config", 'c')
            .setHelpText("absolute path to config-file");

    // debug-flag
    argparser->registerPlain("debug", 'd')
            .setHelpText("enable debug-mode");

    return true;
}

#endif // TSUGUMITESTER_ARGS_H
