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
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use uuid::Uuid;

use ainari_common::error::AinariError;
use ainari_dataset::dataset_io::{DataSetFileReadHandle, DataSetFileWriteHandle};
use ainari_model_parser::model_meta_structs::*;

use crate::core::blocks::core_block::*;
use crate::core::blocks::input_block::*;
use crate::core::blocks::output_block::*;
use crate::core::blocks::start_end_block::*;
use crate::core::blocks::transfer_block::*;
use crate::core::processing::output_buffer::*;
use crate::core::processing::worker_queue::*;

use super::blocks::block_trait::Block;

pub struct InputGroup {
    pub input_values: Vec<f32>,
    pub blocks: Vec<Arc<Mutex<InputBlock>>>,
}

impl InputGroup {
    pub fn new() -> Self {
        InputGroup {
            input_values: Vec::new(),
            blocks: Vec::new(),
        }
    }

    pub fn apply_plain_input(
        &mut self, 
        input_ptr: &[f32],
        allow_creation: bool,
    ) -> Result<(), AinariError> {

        let number_of_blocks = input_ptr.len() / 128 + 1;

        if number_of_blocks > self.blocks.len() {
            // TODO: resize network
        }

        let mut offset = 0usize;
        for i in 0..number_of_blocks {
            let input_block_mutex = &self.blocks[i];
            input_block_mutex.lock().expect("mutex poisoned").apply_input(
                input_ptr,
                offset,
            );
            offset += 128;

            let mut worker_queue = WORKER_QUEUE.lock().expect("mutex poisoned");
            let cycle_number = 0;
            let worker_task = WorkerTask {
                task_type: WorkerTaskType::Train,
                block: Arc::clone(&input_block_mutex) as Arc<Mutex<dyn Block>>,
                cycle_number,
            };
            worker_queue.add(worker_task);
        }

        Ok(())
    }

    fn apply_dataset_to_input(
        &mut self, 
        file_handle: &mut DataSetFileReadHandle,
        cycle_counter: usize,
        allow_creation: bool,
    ) -> Result<(), AinariError> {
        let (input_ptr, _) = file_handle.get_data_from_file(&(cycle_counter as u64))?;
        return self.apply_plain_input(
            input_ptr,
            allow_creation,
        );
    }
}
