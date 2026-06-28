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

use core::result::Result::Err;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

use super::hexagon_block::*;
use super::*;

use crate::core::processing::task_queue::*;
use crate::core::processing::tasks::*;
use crate::database::task_table;

use ainari_common::enums::*;
use ainari_common::error::AinariError;
use ainari_dataset::dataset_io::DataSetFileReadHandle;

// ==================================================================================================

#[derive(Debug, Serialize)]
pub struct StartBlock {
    pub uuid: Uuid,
    pub model_uuid: Uuid,
    #[serde(skip)]
    pub parent_block: Arc<Mutex<HexagonBlock>>,

    pub is_processed: bool,
    #[serde(skip, default = "init_task_handler")]
    pub task_handler_mutex: Arc<Mutex<TaskHandler>>,

    pub block_io: BlockIoBuffer,

    #[serde(skip, default = "init_file_handle")]
    pub file_handle: Option<DataSetFileReadHandle>,
    #[serde(skip, default = "init_task_meta")]
    pub meta: TaskMeta,
    pub task_uuid: Uuid,
    pub task_id: u64,
}

fn init_task_handler() -> Arc<Mutex<TaskHandler>> {
    Arc::new(Mutex::new(TaskHandler::default()))
}

fn init_file_handle() -> Option<DataSetFileReadHandle> {
    None
}

fn init_task_meta() -> TaskMeta {
    TaskMeta::new(0, 0, 0)
}

impl PartialEq for StartBlock {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
            && self.model_uuid == other.model_uuid
            && self.block_io == other.block_io
    }
}

impl StartBlock {
    pub fn new(
        model_uuid: &Uuid,
        parent_block: Arc<Mutex<HexagonBlock>>,
        task_handler_mutex: &Arc<Mutex<TaskHandler>>,
    ) -> Self {
        StartBlock {
            uuid: Uuid::new_v4(),
            model_uuid: *model_uuid,
            parent_block: parent_block,

            is_processed: false,
            task_handler_mutex: task_handler_mutex.clone(),

            block_io: BlockIoBuffer::new(0),

            file_handle: None,
            meta: TaskMeta::new(0, 0, 0),
            task_uuid: Uuid::nil(),
            task_id: 0,
        }
    }

    pub fn init_new_run(
        &mut self,
        task_uuid: &Uuid,
        task_id: u64,
        file_handle: DataSetFileReadHandle,
    ) {
        self.task_uuid = task_uuid.clone();
        self.task_id = task_id;
        self.file_handle = Some(file_handle);
    }

    pub fn get_input_size(&self) -> Result<u64, AinariError> {
        if let Some(file_handle) = &self.file_handle {
            return file_handle.get_col_size();
        } else {
            return Err(AinariError::InternalError(
                "start-block not initialized".to_string(),
            ));
        }
    }
}

impl Block for StartBlock {
    fn process(&mut self) -> Result<bool, AinariError> {
        if !self.is_processed {
            let now = Instant::now();
            if now.duration_since(self.meta.prev_timestamp) >= Duration::from_secs(1) {
                self.meta.prev_timestamp = now;
                let _ = task_table::update_task_progress(
                    &self.task_uuid,
                    &(self.meta.number_of_finished_epochs as i64),
                    &(self.meta.number_of_finished_cycles as i64),
                );

                // check if task was aborted
                if task_table::is_aborted(&self.task_uuid) {
                    return Ok(true);
                }
            }

            // assert_eq!(self.check_input_size(input_ptr), 0);
            // let number_of_blocks = input_ptr.len() / 128 + 1;

            // let mut offset = 0usize;
            // for i in 0..number_of_blocks {
            //     let input_block_mutex = &self.blocks[i];
            //     input_block_mutex
            //         .lock()
            //         .expect("mutex poisoned")
            //         .apply_input(input_ptr, offset);
            //     offset += 128;

            //     let mut worker_queue = WORKER_QUEUE.lock().expect("mutex poisoned");
            //     let cycle_number = 0;
            //     let worker_task = WorkerTask {
            //         block: Arc::clone(&input_block_mutex) as Arc<Mutex<dyn Block>>,
            //     };
            //     worker_queue.add(worker_task);
            // }

            // get task
            let task_handler = self.task_handler_mutex.lock().expect("mutex poisoned");
            let task_mutex = task_handler.get_from_in_progress(&self.task_id)?;
            drop(task_handler);

            // finish task cycle or even epoch
            let mut task = task_mutex.lock().expect("mutex poisoned");
        }

        self.is_processed = true;

        let is_finished = self.block_io.send_forward()?;
        if is_finished {
            self.is_processed = false;
        }
        Ok(is_finished)
    }

    fn get_free_input(&mut self) -> u8 {
        self.block_io.get_free_input()
    }

    /// Gets the UUID of the block.
    ///
    /// # Returns
    ///
    /// The UUID of the block.
    fn get_uuid(&self) -> Uuid {
        self.uuid
    }

    /// Gets the model UUID of the block.
    ///
    /// # Returns
    ///
    /// The model UUID of the block.
    fn get_model_uuid(&self) -> Uuid {
        self.model_uuid
    }

    /// Gets the block I/O buffer.
    ///
    /// # Returns
    ///
    /// A mutable reference to the block I/O buffer.
    fn get_block_io(&mut self) -> &mut BlockIoBuffer {
        &mut self.block_io
    }

    /// Gets the type of the block.
    ///
    /// # Returns
    ///
    /// The type of the block.
    fn get_type(&self) -> ObjectType {
        ObjectType::StartBlock
    }

    fn get_parent_block(&self) -> Option<Arc<Mutex<HexagonBlock>>> {
        Some(self.parent_block.clone())
    }

    /// Serializes the block to a byte vector.
    ///
    /// # Returns
    ///
    /// A byte vector containing the serialized block.
    fn serailize(&self) -> Vec<u8> {
        let cfg = bincode::config::standard();
        bincode::serde::encode_to_vec(self, cfg).expect("Failed to serialize")
    }
}
