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

pub mod db_handle;
pub mod instance_table;
pub mod task_table;

use std::io;

use ainari_common::enums;

/// Initializes the database by setting up required tables and clearing existing instance data.
///
/// This function performs several critical operations:
/// 1. Initializes the instance table in the database.
/// 2. Initializes the task table in the database.
/// 3. Clears all existing instance data from the database to ensure consistency after a restart.
///
/// # Returns
///
/// * `Ok(())` - If all database operations complete successfully.
/// * `Err(Box<dyn std::error::Error>)` - If any database operation fails.
pub fn init_database() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize instance-table
    match instance_table::init_instance_table() {
        Ok(_) => log::info!("Initialized instance-database-table"),
        Err(e) => {
            log::error!("Failed to initialize instance-database-table: {e}");
            return Err(e);
        }
    };
    // Initialize task-table
    match task_table::init_task_table() {
        Ok(_) => log::info!("Initialized task-database-table"),
        Err(e) => {
            log::error!("Failed to initialize task-database-table: {e}");
            return Err(e);
        }
    };

    // Clear all instance from the database. This is necessary because after a restart,
    // all instances are broken and the database doesn't match the real world.
    // To "fix" this issue, all instances have to be removed from the database as well.
    match instance_table::delete_all_instance() {
        Ok(_) => {}
        Err(enums::DbError::InternalError) => {
            let msg = "Error while deleting all instance from DB".to_string();
            log::error!("{msg}");
            let error = io::Error::other(msg);
            return Err(Box::new(error));
        }
        Err(enums::DbError::NotFound) => {
            // Treat NotFound as a recoverable error by returning an empty error message
            let error = io::Error::other("".to_string());
            return Err(Box::new(error));
        }
    }

    Ok(())
}
