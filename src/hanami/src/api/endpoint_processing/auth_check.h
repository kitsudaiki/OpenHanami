#ifndef AUTH_CHECK_H
#define AUTH_CHECK_H

#include <hanami_common/enums.h>
#include <stdint.h>

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

bool validateToken(json& result,
                   const std::string& token,
                   const std::string& endpoint,
                   const HttpRequestType httpType,
                   std::string& errorMessage);

#endif  // AUTH_CHECK_H
