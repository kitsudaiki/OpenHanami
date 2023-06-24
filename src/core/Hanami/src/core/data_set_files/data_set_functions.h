#ifndef DATA_SET_FUNCTIONS_H
#define DATA_SET_FUNCTIONS_H

#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi {
class JsonItem;
}

float* getDataSetPayload(const std::string &location,
                         Kitsunemimi::ErrorContainer &error,
                         const std::string &columnName = "");

bool getDateSetInfo(Kitsunemimi::JsonItem &result,
                    const std::string &dataUuid,
                    const Kitsunemimi::DataMap &context,
                    Kitsunemimi::ErrorContainer &error);

bool getHeaderInformation(Kitsunemimi::JsonItem &result,
                          const std::string &location,
                          Kitsunemimi::ErrorContainer &error);

#endif // DATA_SET_FUNCTIONS_H
