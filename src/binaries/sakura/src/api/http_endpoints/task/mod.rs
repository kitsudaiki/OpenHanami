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

pub mod abort_task_v1_0;
pub mod get_task_v1_0;
pub mod list_task_v1_0;

use std::str::FromStr;

use crate::core::processing::tasks::Task;
use crate::core::processing::worker_handler::*;
use crate::database::task_table;

use ainari_api::errors::ErrorResponse;
use ainari_api_structs::task_structs::{TaskState, TaskType};
use ainari_api_structs::user_context::UserContext;
use ainari_dataset::dataset_io::{DataSetFileReadHandle, read_data_set_file};

/// Converts a string representation of a task type to its enum variant.
///
/// # Arguments
/// * `task_type` - The string to convert
///
/// # Returns
/// * `Result<TaskType, ErrorResponse>` - The converted task type or an error
pub fn convert_task_type(task_type: &String) -> Result<TaskType, ErrorResponse> {
    let converted_task_type = TaskType::from_str(task_type.as_str()).map_err(|_| {
        log::error!("Failed to convert task-type '{task_type}'");
        ErrorResponse::InternalError("Internal Error".to_string())
    })?;

    Ok(converted_task_type)
}

/// Converts a string representation of a task state to its enum variant.
///
/// # Arguments
/// * `task_state` - The string to convert
///
/// # Returns
/// * `Result<TaskState, ErrorResponse>` - The converted task state or an error
pub fn convert_task_state(task_state: &String) -> Result<TaskState, ErrorResponse> {
    let converted_task_state = TaskState::from_str(task_state.as_str()).map_err(|_| {
        log::error!("Failed to convert task-state '{task_state}'");
        ErrorResponse::InternalError("Internal Error".to_string())
    })?;

    Ok(converted_task_state)
}

/// Adds a new task to a model and stores it in the database.
///
/// This function handles both the database storage and the model interface registration
/// of a new task. It ensures the task is properly tracked in both places.
///
/// # Arguments
/// * `task` - The task to add
/// * `task_type` - The type of the task
/// * `context` - The user context for database operations
///
/// # Returns
/// * `Result<(), ErrorResponse>` - Success or an error
pub fn add_task(
    task: Task,
    task_type: &TaskType,
    context: &UserContext,
) -> Result<(), ErrorResponse> {
    task_table::add_new_task(
        &task.uuid,
        &task.resouce_uuid,
        &task.resource_type,
        &task.name,
        task_type,
        context,
    )
    .map_err(|e| {
        log::error!(
            "Failed to add task with UUID '{}' to database with error: {e}.",
            task.uuid
        );
        ErrorResponse::InternalError("Internal Error".to_string())
    })?;

    add_task_to_queue(task);

    Ok(())
}
