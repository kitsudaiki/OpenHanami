/**
 * @file        blossom.h
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

#ifndef HANAMI_LANG_BLOSSOM_H
#define HANAMI_LANG_BLOSSOM_H

#include <hanami_common/functions/time_functions.h>
#include <hanami_common/logger.h>
#include <hanami_common/structs.h>
#include <hanami_common/uuid.h>

class BlossomItem;
class SakuraThread;
class InitialValidator;
class HttpProcessing;

//--------------------------------------------------------------------------------------------------

struct BlossomStatus {
    uint64_t statusCode = OK_RTYPE;
    std::string errorMessage = "";
};

//--------------------------------------------------------------------------------------------------

struct BlossomIO {
    std::string blossomType = "";
    std::string blossomName = "";
    std::string blossomGroupType = "";

    json output;
    json input;

    BlossomIO()
    {
        output = json::object();
        input = json::object();
    }
};

//--------------------------------------------------------------------------------------------------

enum FieldType {
    SAKURA_UNDEFINED_TYPE = 0,
    SAKURA_INT_TYPE = 1,
    SAKURA_FLOAT_TYPE = 2,
    SAKURA_BOOL_TYPE = 3,
    SAKURA_STRING_TYPE = 4,
    SAKURA_ARRAY_TYPE = 5,
    SAKURA_MAP_TYPE = 6
};

//--------------------------------------------------------------------------------------------------

struct FieldDef {
    enum IO_ValueType {
        UNDEFINED_VALUE_TYPE = 0,
        INPUT_TYPE = 1,
        OUTPUT_TYPE = 2,
    };

    const IO_ValueType ioType;
    const FieldType fieldType;
    bool isRequired = true;
    std::string comment = "";
    json match = nullptr;
    json defaultVal = nullptr;
    std::string regex = "";
    long lowerLimit = 0;
    long upperLimit = 0;

    FieldDef(const IO_ValueType ioType, const FieldType fieldType)
        : ioType(ioType), fieldType(fieldType)
    {
    }

    FieldDef& setComment(const std::string& comment)
    {
        this->comment = comment;
        return *this;
    }

    FieldDef& setMatch(const json& match)
    {
        this->match = match;
        return *this;
    }

    FieldDef& setDefault(const json& defaultValue)
    {
        this->defaultVal = defaultValue;
        return *this;
    }

    FieldDef& setRequired(const bool required)
    {
        this->isRequired = required;
        return *this;
    }

    FieldDef& setRegex(const std::string& regex)
    {
        this->regex = regex;
        return *this;
    }

    FieldDef& setLimit(const long lowerLimit, const long upperLimit)
    {
        this->lowerLimit = lowerLimit;
        this->upperLimit = upperLimit;
        return *this;
    }
};

//--------------------------------------------------------------------------------------------------

inline const Hanami::UserContext
convertContext(const json& inputContext)
{
    Hanami::UserContext context;
    context.userId = inputContext["id"];
    context.projectId = inputContext["project_id"];
    context.isAdmin = inputContext["is_admin"];
    context.isProjectAdmin = inputContext["is_project_admin"];
    if (inputContext.contains("token")) {
        context.token = inputContext["token"];
    }

    return context;
}

//--------------------------------------------------------------------------------------------------

class Blossom
{
   public:
    Blossom(const std::string& comment, const bool requiresToken = true);
    virtual ~Blossom();

    const std::string comment;
    std::string tag;
    const bool requiresAuthToken;
    std::vector<HttpResponseTypes> errorCodes = {INTERNAL_SERVER_ERROR_RTYPE};

    const std::map<std::string, FieldDef>* getInputValidationMap() const;
    const std::map<std::string, FieldDef>* getOutputValidationMap() const;

   protected:
    virtual bool runTask(BlossomIO& blossomIO,
                         const json& context,
                         BlossomStatus& status,
                         Hanami::ErrorContainer& error)
        = 0;
    bool allowUnmatched = false;

    FieldDef& registerInputField(const std::string& name, const FieldType fieldType);
    FieldDef& registerOutputField(const std::string& name, const FieldType fieldType);

   private:
    friend SakuraThread;
    friend InitialValidator;
    friend HttpProcessing;

    std::map<std::string, FieldDef> m_inputValidationMap;
    std::map<std::string, FieldDef> m_outputValidationMap;

    bool growBlossom(BlossomIO& blossomIO,
                     const json& context,
                     BlossomStatus& status,
                     Hanami::ErrorContainer& error);
    bool validateFieldsCompleteness(const json& input,
                                    const std::map<std::string, FieldDef>& validationMap,
                                    const FieldDef::IO_ValueType valueType,
                                    std::string& errorMessage);
    void fillDefaultValues(json& values);
    void createError(const BlossomIO& blossomIO,
                     const std::string& errorLocation,
                     Hanami::ErrorContainer& error);
};

#endif  // HANAMI_LANG_BLOSSOM_H
