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

use bytemuck::cast_slice;
use core::result::Result;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::runtime::Builder;
use tokio::task::LocalSet;
use uuid::Uuid;

use ainari_api_structs::task_structs::*;
use ainari_clients::onsen_file_transfer::*;
use ainari_common::error::AinariError;
use ainari_common::secret::Secret;
use ainari_dataset::dataset_io::{DataSetFileReadHandle, DataSetFileWriteHandle};
use ainari_dataset::file_encryption::{decrypt_file, encrypt_file};

use crate::config;
use crate::core::model::model::*;
use crate::database::task_table;

/// Represents the information needed for a training task.
/// Contains input and output dataset handles and a temporary directory path.
#[derive(Debug)]
pub struct TrainInfo {
    pub inputs: HashMap<String, DataSetFileReadHandle>,
    pub outputs: HashMap<String, DataSetFileReadHandle>,
    pub temp_dir: String,
}

/// Represents the information needed for a request task.
/// Contains input dataset handles, a write handle for results, output secret, and a temporary directory path.
#[derive(Debug)]
pub struct RequestInfo {
    pub inputs: HashMap<String, DataSetFileReadHandle>,
    pub results: DataSetFileWriteHandle,
    pub output_secret: Secret,
    pub temp_dir: String,
}

/// Contains information for saving a checkpoint.
/// Includes the Onsen address, file path, and encryption secret.
#[derive(Debug)]
pub struct CheckpointSaveInfo {
    pub onsen_address: String,
    pub file_path: String,
    pub secret: Secret,
}

/// Contains information for restoring a checkpoint.
/// Includes the Onsen address, file path, and decryption secret.
#[derive(Debug)]
pub struct CheckpointRestoreInfo {
    pub onsen_address: String,
    pub file_path: String,
    pub secret: Secret,
}

/// An enumeration of different task variants that a Task can have.
/// Each variant contains different information relevant to that type of task.
#[derive(Debug)]
pub enum TaskVariant {
    Training(TrainInfo),
    Request(Box<RequestInfo>),
    CheckpointSave(CheckpointSaveInfo),
    CheckpointRestore(CheckpointRestoreInfo),
}

/// Metadata for tracking the progress and state of a task.
/// Includes counters for cycles and epochs, timestamps, and completion status.
#[derive(Debug, Clone)]
pub struct TaskMeta {
    pub number_of_cycles: u64,
    pub number_of_epochs: u64,
    pub number_of_finished_cycles: u64,
    pub number_of_finished_epochs: u64,

    // pub forecast_length: u64,
    pub task_cycle_counter: u64,

    pub is_finished: bool,
    pub prev_timestamp: std::time::Instant,
}

impl TaskMeta {
    /// Creates a new TaskMeta instance with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `number_of_cycler_per_epoch` - Total number of cycles per epoch.
    /// * `number_of_epochs` - Total number of epochs.
    /// * `time_length` - Time length for the task in seconds.
    ///
    /// # Returns
    ///
    /// A new TaskMeta instance initialized with the given parameters.
    pub fn new(number_of_cycler_per_epoch: u64, number_of_epochs: u64, _: u64) -> Self {
        Self {
            number_of_cycles: number_of_cycler_per_epoch,
            number_of_epochs,
            number_of_finished_cycles: 0,
            number_of_finished_epochs: 0,
            // forecast_length,
            task_cycle_counter: 0,

            is_finished: false,
            prev_timestamp: std::time::Instant::now(),
        }
    }
}

/// Represents a task that can be executed by the system.
/// Contains a unique identifier, model identifier, task information, and metadata.
#[derive(Debug)]
pub struct Task {
    /// Unique identifier for the task.
    pub uuid: Uuid,
    /// Identifier for the model associated with this task.
    pub model_uuid: Uuid,
    /// Human-readable name for the task.
    #[allow(dead_code)]
    pub name: String,

    /// Variant-specific information for this task.
    pub info: TaskVariant,
    /// Metadata for tracking the progress and state of this task.
    pub meta: TaskMeta,
}

impl Task {
    // ==================================================================================================

    /// Starts the execution of the task.
    ///
    /// # Returns
    ///
    /// `true` if the task should continue execution, `false` if it should pause or stop.
    pub fn start_task(
        &mut self,
        model_uuid: &Uuid,
        model: &Arc<Mutex<Model>>,
    ) -> Result<bool, AinariError> {
        // check if task was aborted
        if task_table::is_aborted(&self.uuid) {
            return Ok(false);
        }

        self.meta.prev_timestamp = Instant::now();
        let _ = task_table::update_task_state(&self.uuid, &TaskState::Active);

        match &mut self.info {
            TaskVariant::Training(task_info) => {
                run_train_task_cycle(&self.uuid, model_uuid, model, &mut self.meta, task_info)?;
                Ok(true)
            }
            TaskVariant::Request(task_info) => {
                run_request_task_cycle(&self.uuid, model_uuid, model, &mut self.meta, task_info)?;
                Ok(true)
            }
            TaskVariant::CheckpointSave(task_info) => {
                handle_checkpoint_save_task(
                    &self.uuid,
                    model_uuid,
                    model,
                    &mut self.meta,
                    task_info,
                )?;
                Ok(false)
            }
            TaskVariant::CheckpointRestore(task_info) => {
                handle_checkpoint_restore_task(
                    &self.uuid,
                    model_uuid,
                    model,
                    &mut self.meta,
                    task_info,
                )?;
                Ok(false)
            }
        }
    }

    /// Finalizes the task, performing cleanup and updating the task state.
    /// For request tasks, it encrypts and uploads the results.
    /// For training tasks, it cleans up temporary files.
    pub fn finalize_task(&mut self) {
        if let TaskVariant::Request(task_info) = &mut self.info {
            let rt = Builder::new_current_thread()
                .enable_all() // I/O & timers
                .build()
                .expect("failed to build runtime");

            // LocalSet allows spawn_local to work
            let local = LocalSet::new();
            let upload_resp = local.block_on(&rt, async {
                encrypt_file(
                    &task_info.results.link.local_file_path,
                    &task_info.results.link.local_encrypted_file_path,
                    &task_info.output_secret,
                )
                .await?;
                upload_file(
                    &task_info.results.link.onsen_address,
                    &task_info.results.link.remote_file_path,
                    &task_info.results.link.local_encrypted_file_path,
                )
                .await
            });

            // delete temp-files
            remove_dir_all(&task_info.temp_dir);

            // handle result
            match upload_resp {
                Ok(()) => {}
                Err(_) => {
                    let _ = task_table::update_task_state(&self.uuid, &TaskState::Error);
                    let _ = task_table::update_task_progress(
                        &self.uuid,
                        &(self.meta.number_of_epochs as i64),
                        &(self.meta.number_of_cycles as i64),
                    );
                    return;
                }
            }
        }

        if let TaskVariant::Training(task_info) = &mut self.info {
            // delete temp-files
            remove_dir_all(&task_info.temp_dir);
        }

        let _ = task_table::update_task_state(&self.uuid, &TaskState::Finished);
        let _ = task_table::update_task_progress(
            &self.uuid,
            &(self.meta.number_of_epochs as i64),
            &(self.meta.number_of_cycles as i64),
        );
    }

    /// Finishes the current cycle of the task and prepares for the next cycle.
    /// Updates progress in the database and checks for task completion.
    pub fn finish_cycle(&mut self, result_data: &Vec<f32>) {
        match &mut self.info {
            TaskVariant::Training(task_info) => {
                finish_train_cycle(&self.uuid, task_info);
            }
            TaskVariant::Request(task_info) => {
                finish_request_cycle(&self.uuid, task_info, result_data);
            }
            _ => {
                return;
            }
        }

        // update current state in database at least after 1 second
        let now = Instant::now();
        if now.duration_since(self.meta.prev_timestamp) >= Duration::from_secs(1) {
            self.meta.prev_timestamp = now;
            let _ = task_table::update_task_progress(
                &self.uuid,
                &(self.meta.number_of_finished_epochs as i64),
                &(self.meta.number_of_finished_cycles as i64),
            );
            if task_table::is_aborted(&self.uuid) {
                self.meta.is_finished = true;
                return;
            }
        }

        // update and check cycle- and epoch-counter
        self.meta.number_of_finished_cycles += 1;
        if self.meta.number_of_finished_cycles >= self.meta.number_of_cycles {
            self.meta.number_of_finished_epochs += 1;
            if self.meta.number_of_finished_epochs == self.meta.number_of_epochs {
                self.meta.is_finished = true;
                return;
            } else {
                self.meta.number_of_finished_cycles = 0;
            }
        }
        self.meta.task_cycle_counter += 1;
    }

    /// Checks if the task has been completed.
    ///
    /// # Returns
    ///
    /// `true` if the task is finished, `false` otherwise.
    pub fn is_task_finished(&self) -> bool {
        self.meta.is_finished
    }
}

/// Finalizes a training cycle. Currently an empty implementation.
///
/// # Arguments
///
/// * `_` - The UUID of the task.
/// * `_` - Mutable reference to the training information.
fn finish_train_cycle(_: &Uuid, _: &mut TrainInfo) {}

/// Finalizes a request cycle by writing the output to the dataset.
///
/// # Arguments
///
/// * `task_uuid` - The UUID of the task.
/// * `model_uuid` - The UUID of the model associated with the task.
/// * `task_info` - Mutable reference to the request information.
fn finish_request_cycle(task_uuid: &Uuid, task_info: &mut RequestInfo, result_data: &Vec<f32>) {
    // get output-values form backend and write them into the dataset
    match write_output_into_dataset(&mut task_info.results, result_data) {
        Ok(()) => {}
        Err(AinariError::Unauthorized(msg)) => {
            let _ = task_table::set_error_state(task_uuid, &msg);
        }
        Err(AinariError::InvalidInput(msg)) => {
            let _ = task_table::set_error_state(task_uuid, &msg);
        }
        Err(AinariError::InternalError(msg)) => {
            log::error!("Error while writing output into dataset: {msg}");
            let db_msg = "internal error".to_string();
            let _ = task_table::set_error_state(task_uuid, &db_msg);
        }
    }
}

/// Writes model output data into a dataset file.
///
/// This function reads output data from the model's output buffers and writes it to the specified
/// dataset file. It processes each hexagon's output data according to the dataset's column description.
///
/// # Arguments
///
/// * `model_uuid` - Unique identifier for the model
/// * `file_handle` - Mutable reference to the dataset file handle for writing
///
/// # Returns
///
/// * `Result<(), AinariError>` - Returns Ok(()) on success, or an AinariError on failure
fn write_output_into_dataset(
    file_handle: &mut DataSetFileWriteHandle,
    result_data: &Vec<f32>,
) -> Result<(), AinariError> {
    let output_bytes = cast_slice(&result_data);
    let _ = file_handle.target_file.write_all(output_bytes);
    // TODO: error-handling while write_all
    Ok(())
}

/// Executes a single training cycle for a task.
///
/// This function updates the task progress in the database, checks for task abortion,
/// and applies input and output datasets to the model.
///
/// # Arguments
///
/// * `task_uuid` - Unique identifier for the task
/// * `model_uuid` - Unique identifier for the model
/// * `meta` - Mutable reference to task metadata containing progress information
/// * `task_info` - Mutable reference to training information containing input/output datasets
fn run_train_task_cycle(
    task_uuid: &Uuid,
    _: &Uuid,
    model_mutex: &Arc<Mutex<Model>>,
    meta: &mut TaskMeta,
    task_info: &mut TrainInfo,
) -> Result<(), AinariError> {
    // update current state in database at least after 1 second
    let now = Instant::now();
    if now.duration_since(meta.prev_timestamp) >= Duration::from_secs(1) {
        meta.prev_timestamp = now;
        let _ = task_table::update_task_progress(
            task_uuid,
            &(meta.number_of_finished_epochs as i64),
            &(meta.number_of_finished_cycles as i64),
        );

        // check if task was aborted
        if task_table::is_aborted(task_uuid) {
            return Ok(());
        }
    }

    let model = model_mutex.lock().expect("mutex poisoned");

    for (_, file_handle) in task_info.outputs.drain() {
        let end_block_mutex = model.get_end_block();
        let mut end_block = end_block_mutex.lock().expect("mutex poisoned");
        let task_id = 0;
        end_block.init_new_train_run(task_uuid, task_id, file_handle);
    }

    for (hexagon_name, file_handle) in task_info.inputs.drain() {
        let start_block_mutex = model.get_start_block(&hexagon_name)?;
        let mut start_block = start_block_mutex.lock().expect("mutex poisoned");
        let task_id = 0u64;
        start_block.init_new_run(task_uuid, task_id, file_handle);
    }

    Ok(())
}

/// Executes a single processing cycle for a request task.
///
/// This function updates the task progress in the database, checks for task abortion,
/// and applies input datasets to the model.
///
/// # Arguments
///
/// * `task_uuid` - Unique identifier for the task
/// * `model_uuid` - Unique identifier for the model
/// * `meta` - Mutable reference to task metadata containing progress information
/// * `task_info` - Mutable reference to request information containing input datasets
fn run_request_task_cycle(
    task_uuid: &Uuid,
    _: &Uuid,
    model_mutex: &Arc<Mutex<Model>>,
    meta: &mut TaskMeta,
    task_info: &mut RequestInfo,
) -> Result<(), AinariError> {
    // update current state in database at least after 1 second
    let now = Instant::now();
    if now.duration_since(meta.prev_timestamp) >= Duration::from_secs(1) {
        meta.prev_timestamp = now;
        let _ = task_table::update_task_progress(
            task_uuid,
            &0,
            &(meta.number_of_finished_cycles as i64),
        );

        // check if task was aborted
        if task_table::is_aborted(task_uuid) {
            return Ok(());
        }
    }

    let model = model_mutex.lock().expect("mutex poisoned");

    let end_block_mutex = model.get_end_block();
    let mut end_block = end_block_mutex.lock().expect("mutex poisoned");
    let task_id = 0;
    end_block.init_new_request_run(task_uuid, task_id);

    for (hexagon_name, file_handle) in task_info.inputs.drain() {
        let start_block_mutex = model.get_start_block(&hexagon_name)?;
        let mut start_block = start_block_mutex.lock().expect("mutex poisoned");
        let task_id = 0;
        start_block.init_new_run(task_uuid, task_id, file_handle);
    }

    Ok(())
}

/// Handles the task of saving a model checkpoint.
///
/// This function creates a checkpoint of the model, encrypts it, and uploads it to the specified
/// storage location. It manages temporary files and updates the task state in the database.
///
/// # Arguments
///
/// * `task_uuid` - Unique identifier for the task
/// * `model_uuid` - Unique identifier for the model
/// * `_` - Unused TaskMeta parameter (kept for interface consistency)
/// * `task_info` - Mutable reference to checkpoint save information containing storage details
fn handle_checkpoint_save_task(
    task_uuid: &Uuid,
    model_uuid: &Uuid,
    _: &Arc<Mutex<Model>>,
    _: &mut TaskMeta,
    task_info: &mut CheckpointSaveInfo,
) -> Result<(), AinariError> {
    // create file-paths for temporary files
    let local_temp_file_path = format!(
        "{}/{}",
        config::CONFIG.storage.tempfile_location,
        model_uuid
    );
    let local_encrypted_temp_file_path = format!("{local_temp_file_path}_encrypted");

    {
        // match model_handler.create_checkpoint(model_uuid, &local_temp_file_path) {
        //     Ok(()) => {}
        //     Err(_) => {
        //         let _ = fs::remove_file(&local_temp_file_path);
        //         let _ = task_table::update_task_state(task_uuid, &TaskState::Error);
        //         let _ = task_table::update_task_progress(task_uuid, &1, &1);
        //         return;
        //     }
        // }

        // Create a single-threaded runtime
        let rt = Builder::new_current_thread()
            .enable_all() // I/O & timers
            .build()
            .expect("failed to build runtime");

        // LocalSet allows spawn_local to work
        let local = LocalSet::new();
        let upload_resp = local.block_on(&rt, async {
            encrypt_file(
                &local_temp_file_path,
                &local_encrypted_temp_file_path,
                &task_info.secret,
            )
            .await?;
            upload_file(
                &task_info.onsen_address,
                &task_info.file_path,
                &local_encrypted_temp_file_path,
            )
            .await
        });

        match upload_resp {
            Ok(()) => {}
            Err(_) => {
                let _ = task_table::update_task_state(task_uuid, &TaskState::Error);
                let _ = task_table::update_task_progress(task_uuid, &1, &1);
                return Ok(());
            }
        }

        let _ = task_table::update_task_state(task_uuid, &TaskState::Finished);
        let _ = task_table::update_task_progress(task_uuid, &1, &1);
    }

    let _ = fs::remove_file(&local_temp_file_path);
    let _ = fs::remove_file(&local_encrypted_temp_file_path);

    Ok(())
}

/// Handles the task of restoring a model from a checkpoint.
///
/// This function downloads an encrypted checkpoint file, decrypts it, and restores the model from
/// the checkpoint. It manages temporary files and updates the task state in the database.
///
/// # Arguments
///
/// * `task_uuid` - Unique identifier for the task
/// * `model_uuid` - Unique identifier for the model
/// * `_` - Unused TaskMeta parameter (kept for interface consistency)
/// * `task_info` - Mutable reference to checkpoint restore information containing storage details
fn handle_checkpoint_restore_task(
    task_uuid: &Uuid,
    model_uuid: &Uuid,
    _: &Arc<Mutex<Model>>,
    _: &mut TaskMeta,
    task_info: &mut CheckpointRestoreInfo,
) -> Result<(), AinariError> {
    // create file-paths for temporary files
    let local_temp_file_path = format!(
        "{}/{}",
        config::CONFIG.storage.tempfile_location,
        model_uuid
    );
    let local_encrypted_temp_file_path = format!("{local_temp_file_path}_encrypted");

    {
        // Create a single-threaded runtime
        let rt = Builder::new_current_thread()
            .enable_all() // I/O & timers
            .build()
            .expect("failed to build runtime");

        // LocalSet allows spawn_local to work
        let local = LocalSet::new();
        let download_resp = local.block_on(&rt, async {
            let resp = download_file(
                &task_info.onsen_address,
                &task_info.file_path,
                &local_encrypted_temp_file_path,
            )
            .await;
            decrypt_file(
                &local_encrypted_temp_file_path,
                &local_temp_file_path,
                &task_info.secret,
            )
            .await?;

            resp
        });

        match download_resp {
            Ok(()) => {}
            Err(e) => {
                log::error!("Error in checkpoint-restore-task: {e}");
                let _ = task_table::update_task_state(task_uuid, &TaskState::Error);
                let _ = task_table::update_task_progress(task_uuid, &1, &1);
                return Ok(());
            }
        }

        // restore model from the downloaded and decrypted checkpoint-file
        // match model_handler.restore_checkpoint(model_uuid, &local_temp_file_path) {
        //     Ok(()) => {}
        //     Err(_) => {
        //         let _ = task_table::update_task_state(task_uuid, &TaskState::Error);
        //         let _ = task_table::update_task_progress(task_uuid, &1, &1);
        //         return;
        //     }
        // }

        // delete temporary checkpoint-file
        let _ = task_table::update_task_state(task_uuid, &TaskState::Finished);
        let _ = task_table::update_task_progress(task_uuid, &1, &1);
    }

    // cleanup temp-files
    let _ = fs::remove_file(&local_temp_file_path);
    let _ = fs::remove_file(&local_encrypted_temp_file_path);

    Ok(())
}

/// Removes a directory and all its contents from the filesystem.
///
/// This function attempts to delete a directory and all files within it. If the operation fails,
/// it logs an error message but does not propagate the error.
///
/// # Arguments
///
/// * `target_dir_path` - Path to the directory to be removed
fn remove_dir_all(target_dir_path: &String) {
    // delete all temporary files
    let _ = std::fs::remove_dir_all(target_dir_path).map_err(|e| {
        log::error!("Failed to delete temp-dir {target_dir_path} from disk with error {e}.");
    });
}
