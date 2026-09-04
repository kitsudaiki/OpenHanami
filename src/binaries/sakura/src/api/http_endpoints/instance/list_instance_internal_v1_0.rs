// Copyright 2022-2026 Tobias Anker <tobias.anker@kitsunemimi.moe>

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use actix_web::web::Json;
use apistos::api_operation;

use crate::database::instance_table;

use ainari_api::common_functions::convert_uuid;
use ainari_api::errors::ErrorResponse;
use ainari_api_structs::instance_structs::*;
use ainari_api_structs::user_context::UserContext;

#[api_operation(
    tag = "instance",
    summary = "List instance",
    description = r###"List basic information of all instance from the database."###,
    error_code = 401,
    error_code = 500
)]
pub async fn list_instance_internal(
    context: UserContext,
) -> Result<Json<InstanceListResp>, ErrorResponse> {
    let instances = match instance_table::list_instances(&context) {
        Ok(instances) => instances,
        Err(e) => {
            log::error!("Failed to get list of instances form database: '{e}'");
            return Err(ErrorResponse::InternalError("Internal Error".to_string()));
        }
    };

    let mut resp = InstanceListResp { instances: Vec::new() };

    for instance in instances {
        let uuid = convert_uuid(&instance.uuid)?;
        let obj = InstanceBasicResp {
            uuid,
            name: instance.name,
            proxy_port: 0,
        };

        resp.instances.push(obj);
    }

    Ok(Json(resp))
}
