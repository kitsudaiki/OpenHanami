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
use actix_web::web::Path;
use apistos::api_operation;
use uuid::Uuid;

use crate::config;
use crate::database::host_table;
use crate::database::meta_instance_table;

use ainari_api::common_functions::*;
use ainari_api::errors::ErrorResponse;
use ainari_api_structs::instance_structs::*;
use ainari_api_structs::user_context::UserContext;
use ainari_clients::endpoints::*;
use ainari_clients::instance as instance_clients;
use ainari_clients::proxy as proxy_clients;

#[api_operation(
    tag = "instance",
    summary = "Get instance",
    description = r###"Get information of a instance from the database."###,
    error_code = 400,
    error_code = 401,
    error_code = 404,
    error_code = 500
)]
pub async fn get_instance(
    instance_uuid: Path<Uuid>,
    context: UserContext,
) -> Result<Json<InstanceResp>, ErrorResponse> {
    let instance_data = meta_instance_table::get_meta_instance(&instance_uuid, &context)
        .map_err(|e| map_db_uuid_get_delete_error("instance-meta", &instance_uuid, e))?;

    let sakura_uuid = convert_uuid(&instance_data.sakura_host_uuid)?;

    let host_data = host_table::get_host(&sakura_uuid, &context)
        .map_err(|e| map_db_uuid_get_delete_error("sakura-host", &sakura_uuid, e))?;

    // get endpoints from miko
    let miko_endpoint = &config::CONFIG.miko;
    let endpoints = get_endpoints(miko_endpoint, config::CONFIG.skip_tls_verification)
        .await
        .map_err(map_ainari_error_to_api_response)?;

    // send request to torii to get port
    let proxy_uuid = convert_uuid(&instance_data.proxy_uuid)?;
    let proxy_resp = proxy_clients::get_proxy(
        &endpoints.torii,
        &context.token,
        &proxy_uuid,
        config::CONFIG.skip_tls_verification,
    )
    .await
    .map_err(map_ainari_error_to_api_response)?;

    // get instance-information from sakura-host
    let mut instance_resp = instance_clients::get_instance(
        &host_data.address,
        &context.token,
        &config::INTERNAL_API_KEY,
        &instance_uuid,
        config::CONFIG.skip_tls_verification,
    )
    .await
    .map_err(map_ainari_error_to_api_response)?;

    // set port in response
    instance_resp.torii_port = proxy_resp.port;

    Ok(Json(instance_resp))
}
