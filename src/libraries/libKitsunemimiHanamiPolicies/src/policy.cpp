/**
 * @file        policy.cpp
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

#include <libKitsunemimiHanamiPolicies/policy.h>

#include <libKitsunemimiCommon/items/data_items.h>
#include <policy_parsing/policy_parser_interface.h>

namespace Kitsunemimi::Hanami
{

/**
 * @brief Policy::Policy
 */
Policy::Policy() {}

/**
 * @brief Policy::~Policy
 */
Policy::~Policy() {}

/**
 * @brief parse policy-file content
 *
 * @param input input-string with policy-definition to parse
 * @param error reference for error-output
 *
 * @return true, if parsing was successful, else false
 */
bool
Policy::parse(const std::string &input,
              ErrorContainer &error)
{
    if(input.size() == 0) {
        return false;
    }

    PolicyParserInterface* parser = PolicyParserInterface::getInstance();
    return parser->parse(&m_policyRules, input, error);
}

/**
 * @brief check if request is allowed by policy
 *
 * @param endpoint requested endpoint of the component
 * @param type http-request-type
 * @param role role which has to be checked
 *
 * @return true, if check was successfully, else false
 */
bool
Policy::checkUserAgainstPolicy(const std::string &endpoint,
                               const HttpRequestType type,
                               const std::string &role)
{
    std::map<std::string, PolicyEntry>::const_iterator endpoint_it;
    endpoint_it = m_policyRules.find(endpoint);

    if(endpoint_it != m_policyRules.end()) {
        return checkEntry(endpoint_it->second,  type, role);
    }

    return false;
}

/**
 * @brief check role against policy-set for a specific endpoint
 *
 * @param entry policy-entry with all predefined rules of a specific endpoint
 * @param type http-request-type
 * @param role role which has to be checked
 *
 * @return true, if a rule in the entry match the role of the user
 */
bool
Policy::checkEntry(const PolicyEntry &entry,
                   const HttpRequestType type,
                   const std::string &role)
{
    switch(type)
    {
        case GET_TYPE:
            return checkRuleList(entry.getRules, role);
            break;
        case POST_TYPE:
            return checkRuleList(entry.postRules, role);
            break;
        case PUT_TYPE:
            return checkRuleList(entry.putRules, role);
            break;
        case DELETE_TYPE:
            return checkRuleList(entry.deleteRules, role);
            break;
        case HEAD_TYPE:
            break;
        case UNKNOWN_HTTP_TYPE:
            break;
    }

    return false;
}

/**
 * @brief check against the role-list
 *
 * @param rules list of rules of the requested endpoint
 * @param role role which has to be checked
 *
 * @return true, if user-role is in the list of rules, else false
 */
bool
Policy::checkRuleList(const std::vector<std::string> &rules,
                      const std::string &role)
{
    for(const std::string& r : rules)
    {
        if(r == role) {
            return true;
        }
    }

    return false;
}

}
