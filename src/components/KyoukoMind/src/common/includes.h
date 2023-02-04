/**
 * @file        includes.h
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

#ifndef KYOUKOMIND_INCLUDES_H
#define KYOUKOMIND_INCLUDES_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>

#include <vector>
#include <string.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <map>
#include <assert.h>
#include <thread>
#include <mutex>
#include <istream>
#include <time.h>
#include <queue>
#include <condition_variable>
#include <unistd.h>
#include <math.h>
#include <cmath>
#include <utility>
#include <atomic>
#include <uuid/uuid.h>

#include <libKitsunemimiCommon/buffer/data_buffer.h>
using Kitsunemimi::DataBuffer;

#include <libKitsunemimiCommon/buffer/stack_buffer.h>
using Kitsunemimi::StackBuffer;

#include <libKitsunemimiCommon/items/data_items.h>
using Kitsunemimi::DataItem;
using Kitsunemimi::DataArray;
using Kitsunemimi::DataValue;
using Kitsunemimi::DataMap;

#include <libKitsunemimiJson/json_item.h>
using Kitsunemimi::JsonItem;

#include <libKitsunemimiHanamiCommon/uuid.h>

#endif // KYOUKOMIND_INCLUDES_H
