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

use std::net::ToSocketAddrs;

use actix_web::web::Json;
use apistos::actix::CreatedJson;
use apistos::api_operation;
use uuid::Uuid;
use validator::Validate;

use crate::database::instance_table;

use ainari_api::common_functions::*;
use ainari_api::errors::ErrorResponse;
use ainari_api_structs::instance_structs::*;
use ainari_api_structs::user_context::UserContext;

#[api_operation(
    tag = "instance",
    summary = "Create new instance",
    description = r###"Create new instance based on a instance-template."###,
    error_code = 400,
    error_code = 401,
    error_code = 500
)]
pub async fn create_instance_internal(
    body: Json<InstanceCreateReq>,
    context: UserContext,
) -> Result<CreatedJson<InstanceResp>, ErrorResponse> {
    // validate incoming json
    body.validate()
        .map_err(|e| ErrorResponse::BadRequest(format!("Invalid input: {e}")))?;

    let instance_uuid = Uuid::new_v4();

    // // add new instance to database
    // match instance_table::add_new_instance(
    //     &instance_uuid,
    //     &body.name,
    //     &body.template,
    //     &inputs,
    //     &outputs,
    //     &context,
    // ) {
    //     Ok(_) => {}
    //     Err(e) => {
    //         let msg =
    //             format!("Failed to add instance with UUID '{instance_uuid}' to database with error: {e}");
    //         log::error!("{msg}");
    //         return Err(ErrorResponse::InternalError("Internal Error".to_string()));
    //     }
    // };

    // get new created instance from database to get addtional information
    let instance_data = instance_table::get_instance(&instance_uuid, &context)
        .map_err(|e| map_db_uuid_get_delete_error("instance", &instance_uuid, e))?;

    let resp = InstanceResp {
        uuid: instance_uuid,
        name: instance_data.name,
        template: "".to_string(),
        torii_port: 0,
        created_by: instance_data.created_by,
        created_at: instance_data.created_at,
        updated_by: instance_data.updated_by,
        updated_at: instance_data.updated_at,
    };

    Ok(CreatedJson(resp))
}
