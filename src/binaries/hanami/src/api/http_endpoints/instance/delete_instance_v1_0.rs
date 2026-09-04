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

use actix_web::web::Path;
use apistos::actix::NoContent;
use apistos::api_operation;
use uuid::Uuid;

use crate::config;
use crate::database::host_table;
use crate::database::meta_instance_table;

use ainari_api::common_functions::*;
use ainari_api::errors::ErrorResponse;
use ainari_api_structs::user_context::UserContext;
use ainari_clients::endpoints::get_endpoints;
use ainari_clients::instance as instance_clients;
use ainari_clients::proxy as proxy_clients;

#[api_operation(
    tag = "instance",
    summary = "Delete instance",
    description = r###"Delete a instance from the database and core."###,
    error_code = 400,
    error_code = 401,
    error_code = 404,
    error_code = 500
)]
pub async fn delete_instance(
    instance_uuid: Path<Uuid>,
    context: UserContext,
) -> Result<NoContent, ErrorResponse> {
    let instance_data = meta_instance_table::get_meta_instance(&instance_uuid, &context)
        .map_err(|e| map_db_uuid_get_delete_error("instance-meta", &instance_uuid, e))?;

    let sakura_uuid = convert_uuid(&instance_data.sakura_host_uuid)?;
    let proxy_uuid = convert_uuid(&instance_data.proxy_uuid)?;

    let host_data = host_table::get_host(&sakura_uuid, &context)
        .map_err(|e| map_db_uuid_get_delete_error("sakura-host", &sakura_uuid, e))?;

    let endpoints = get_endpoints(&config::CONFIG.miko, config::CONFIG.skip_tls_verification)
        .await
        .map_err(map_ainari_error_to_api_response)?;

    // send request to sakura to delete the instance
    instance_clients::delete_instance(
        &host_data.address,
        &context.token,
        &config::INTERNAL_API_KEY,
        &instance_uuid,
        config::CONFIG.skip_tls_verification,
    )
    .await
    .map_err(map_ainari_error_to_api_response)?;

    // delete instance from database of hanami
    meta_instance_table::delete_meta_instance(&instance_uuid, &context)
        .map_err(|e| map_db_uuid_get_delete_error("instance-meta", &instance_uuid, e))?;

    // send request to torii to delete the proxy, which is connected to the instance
    proxy_clients::delete_proxy(
        &endpoints.torii,
        &context.token,
        &config::INTERNAL_API_KEY,
        &proxy_uuid,
        config::CONFIG.skip_tls_verification,
    )
    .await
    .map_err(map_ainari_error_to_api_response)?;

    Ok(NoContent)
}
