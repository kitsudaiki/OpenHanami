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

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use super::hexagon_block::*;
use super::*;

use crate::core::processing::task_queue::*;
use crate::core::processing::tasks::*;

use ainari_common::enums::*;
use ainari_common::error::AinariError;
use ainari_dataset::dataset_io::{DataSetFileReadHandle, DataSetFileWriteHandle};

// ==================================================================================================

#[derive(Debug, Serialize)]
pub struct EndBlock {
    pub uuid: Uuid,
    pub model_uuid: Uuid,
    #[serde(skip)]
    pub parent_block: Arc<Mutex<HexagonBlock>>,

    pub is_processed: bool,
    #[serde(skip, default = "init_task_handler")]
    pub task_handler_mutex: Arc<Mutex<TaskHandler>>,

    pub block_io: BlockIoBuffer,

    pub output_values: Vec<f32>,

    #[serde(skip, default = "init_read_file_handle")]
    pub read_file_handle: Option<DataSetFileReadHandle>,
    #[serde(skip, default = "init_task_meta")]
    pub meta: TaskMeta,
    pub task_uuid: Uuid,
    pub task_id: u64,
}

fn init_task_handler() -> Arc<Mutex<TaskHandler>> {
    Arc::new(Mutex::new(TaskHandler::default()))
}

fn init_read_file_handle() -> Option<DataSetFileReadHandle> {
    None
}

fn init_task_meta() -> TaskMeta {
    TaskMeta::new(0, 0, 0)
}

impl PartialEq for EndBlock {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
            && self.model_uuid == other.model_uuid
            && self.block_io == other.block_io
    }
}

impl EndBlock {
    pub fn new(
        model_uuid: &Uuid,
        parent_block: Arc<Mutex<HexagonBlock>>,
        task_handler_mutex: &Arc<Mutex<TaskHandler>>,
        number_of_inputs: usize,
    ) -> Self {
        EndBlock {
            uuid: Uuid::new_v4(),
            model_uuid: *model_uuid,
            parent_block: parent_block,

            is_processed: false,
            task_handler_mutex: task_handler_mutex.clone(),

            block_io: BlockIoBuffer::new(number_of_inputs),

            output_values: Vec::new(),

            read_file_handle: None,
            meta: TaskMeta::new(0, 0, 0),
            task_uuid: Uuid::nil(),
            task_id: 0,
        }
    }

    pub fn init_new_train_run(
        &mut self,
        task_uuid: &Uuid,
        task_id: u64,
        file_handle: DataSetFileReadHandle,
    ) {
        self.task_uuid = task_uuid.clone();
        self.task_id = task_id;
        self.read_file_handle = Some(file_handle);
    }

    pub fn init_new_request_run(&mut self, task_uuid: &Uuid, task_id: u64) {
        self.task_uuid = task_uuid.clone();
        self.task_id = task_id;
    }
}

impl Block for EndBlock {
    fn process(&mut self) -> Result<bool, AinariError> {
        let task_id: u64;
        // let cycle_number: u64;
        // let do_train: bool;

        // get task-specific information from the input-buffer
        if let Some(first) = self.block_io.input_buffer.first() {
            task_id = first.task_id;
            // cycle_number = first.cycle_number;
            // do_train = first.do_train;
        } else {
            // TODO: error-handling, because this case should never appear
            return Ok(false);
        }

        // get task
        let task_handler = self.task_handler_mutex.lock().expect("mutex poisoned");
        let task_mutex = task_handler.get_from_in_progress(&task_id)?;
        drop(task_handler);

        // finish task cycle or even epoch
        let mut task = task_mutex.lock().expect("mutex poisoned");
        task.finish_cycle(&self.output_values);

        // self.output_values.resize(number_of_outputs_copy, 0.0f32);
        // self.expected_values.resize(number_of_outputs_copy, 0.0f32);

        Ok(true)
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
        ObjectType::EndBlock
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
