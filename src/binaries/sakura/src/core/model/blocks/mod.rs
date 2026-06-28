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

pub mod core_block;
pub mod end_block;
pub mod hexagon_block;
pub mod input_block;
pub mod output_block;
pub mod start_block;

// ==================================================================================================

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use ainari_common::constants::*;
use ainari_common::enums::*;
use ainari_common::error::AinariError;

use crate::core::processing::worker_queue::*;
use hexagon_block::*;

// ==================================================================================================

pub trait Block: Send + Sync + Debug {
    fn process(&mut self) -> Result<bool, AinariError>;

    fn get_free_input(&mut self) -> u8;

    fn get_uuid(&self) -> Uuid;
    fn get_model_uuid(&self) -> Uuid;
    fn get_parent_block(&self) -> Option<Arc<Mutex<HexagonBlock>>>;
    #[allow(dead_code)]
    fn get_type(&self) -> ObjectType;
    #[allow(dead_code)]
    fn serailize(&self) -> Vec<u8>;

    fn get_block_io(&mut self) -> &mut BlockIoBuffer;
}

// ==================================================================================================

/// Structure representing data for axons in a neural block.
///
/// Contains an array of f32 values representing axon data, serialized with BigArray for efficient storage.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxonData {
    #[serde(with = "BigArray")]
    pub axons: [f32; BLOCK_DIM],
}

impl AxonData {
    pub fn default() -> Self {
        AxonData {
            axons: std::array::from_fn(|_| 0.0f32),
        }
    }
}

// ==================================================================================================

/// Structure representing a section of axons with associated metadata and state.
///
/// This structure contains axon data, target information, and state flags for processing in a neural network.
#[derive(Debug, Serialize, Deserialize)]
pub struct AxonSection {
    pub data: AxonData,
    model_uuid: Uuid,

    /// Flag indicating whether this axon section has already been sent to its target.
    is_already_send: bool,
    /// Flag indicating whether this axon section is ready to receive new input.
    is_ready_for_new_input: bool,

    pub task_id: u64,
    pub cycle_number: f64,
    pub do_train: bool,

    target_block_uuid: Uuid,
    target_pos: u8,
    target_type: ObjectType,

    /// Reference to the target block where this axon section will be sent.
    #[serde(skip)]
    target_block: Option<Arc<Mutex<dyn Block>>>,
}

impl Clone for AxonSection {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            model_uuid: self.model_uuid,

            is_already_send: self.is_already_send,
            is_ready_for_new_input: self.is_ready_for_new_input,

            task_id: self.task_id,
            cycle_number: self.cycle_number,
            do_train: self.do_train,

            target_block_uuid: self.target_block_uuid,
            target_pos: self.target_pos,
            target_type: self.target_type.clone(),

            target_block: self.target_block.clone(),
        }
    }
}

impl AxonSection {
    pub fn default() -> Self {
        AxonSection {
            data: AxonData::default(),
            model_uuid: Uuid::nil(),
            task_id: 0,
            cycle_number: 0f64,
            do_train: false,
            target_block_uuid: Uuid::nil(),
            target_pos: UNINIT_STATE_8,
            target_block: None,
            target_type: ObjectType::Unknown,
            is_already_send: false,
            is_ready_for_new_input: true,
        }
    }
}

impl PartialEq for AxonSection {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.model_uuid == other.model_uuid
            && self.task_id == other.task_id
            && self.cycle_number == other.cycle_number
            && self.do_train == other.do_train
            && self.target_block_uuid == other.target_block_uuid
            && self.target_pos == other.target_pos
            && self.is_already_send == other.is_already_send
            && self.is_ready_for_new_input == other.is_ready_for_new_input
    }
}

// ==================================================================================================

/// A buffer structure for managing input and output axon sections in a neural block.
///
/// This structure maintains buffers for both input and output axon sections, along with
/// counters to track their usage and state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockIoBuffer {
    pub input_buffer: Vec<AxonSection>,
    pub output_buffer: Vec<AxonSection>,

    /// Total number of inputs currently being connected
    pub inputs_in_use: u8,

    pub task_id: u64,
    pub cycle_number: f64,
    pub do_train: bool,
}

impl BlockIoBuffer {
    /// Creates a new BlockIoBuffer with the specified number of input slots.
    pub fn new(number_of_inputs: usize) -> Self {
        let mut new_buffer = BlockIoBuffer {
            inputs_in_use: 0,

            input_buffer: Vec::new(),
            output_buffer: Vec::new(),

            task_id: 0,
            cycle_number: 0f64,
            do_train: false,
        };
        new_buffer
            .input_buffer
            .resize_with(number_of_inputs, AxonSection::default);

        new_buffer
    }

    /// Sends the output axon sections to their target blocks and schedules processing tasks.
    pub fn send_forward(&mut self) -> Result<bool, AinariError> {
        let mut fully_processed = true;

        // send outputs to target
        for output_axon_section in self.output_buffer.iter_mut() {
            if output_axon_section.is_already_send {
                continue;
            }

            // Get the target block mutex or skip this axon section
            let target_block_mutex = if let Some(t) = &output_axon_section.target_block {
                t
            } else {
                continue;
            };

            // Lock the target block and update its input buffer
            let mut target_block = target_block_mutex.lock().expect("mutex poisoned");
            let target_bock_io = target_block.get_block_io();
            let target_pos = output_axon_section.target_pos as usize;

            // check if the target-block is ready to receive new input
            if !target_bock_io.input_buffer[target_pos].is_ready_for_new_input {
                fully_processed = false; // try again later
                continue;
            }

            // set values to forward
            output_axon_section.task_id = self.task_id;
            output_axon_section.cycle_number = self.cycle_number;
            output_axon_section.do_train = self.do_train;
            output_axon_section.is_ready_for_new_input = false;

            target_bock_io.input_buffer[target_pos] = output_axon_section.clone();

            output_axon_section.is_already_send = true;

            // Check if all required inputs are present and schedule a worker task if so
            if target_bock_io.is_input_complete() {
                // Add the task to the worker queue
                let mut worker_queue = WORKER_QUEUE.lock().expect("mutex poisoned");
                let worker_task = WorkerTask {
                    block: Arc::clone(target_block_mutex),
                };
                worker_queue.add(worker_task);
            }
        }

        // mark input as ready, so the
        if fully_processed {
            self.reset_self_block_io();
        }

        Ok(fully_processed)
    }

    /// Gets the index of a free input slot in the buffer.
    ///
    /// Returns 255 if no input slots are available.
    pub fn get_free_input(&mut self) -> u8 {
        // check if inputs are available
        if self.inputs_in_use as usize == self.input_buffer.len() {
            // No input slots available
            return 255;
        }

        // update io-buffer
        let current_pos = self.inputs_in_use;
        self.input_buffer[current_pos as usize] = AxonSection::default();
        self.inputs_in_use += 1;

        current_pos
    }

    /// Resets the state of all input and output axon sections in the buffer.
    ///
    /// Sets is_already_send to false for output sections and is_ready_for_new_input to true
    /// for input sections.
    fn reset_self_block_io(&mut self) {
        for output_buffer in self.output_buffer.iter_mut() {
            output_buffer.is_already_send = false;
        }
        for input_buffer in self.input_buffer.iter_mut() {
            input_buffer.is_ready_for_new_input = true;
        }
    }

    /// Checks if all required inputs have been received and are ready for processing.
    fn is_input_complete(&self) -> bool {
        for input_buffer in self.input_buffer.iter().take(self.inputs_in_use as usize) {
            if input_buffer.is_ready_for_new_input == true {
                return false;
            }
        }

        return true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::hexagon_block::*;
    use super::super::output_block::*;

    #[test]
    fn test_send_forward() {
        // create blocks for testing
        let model_uuid = Uuid::new_v4();
        let parent_block = Arc::new(Mutex::new(HexagonBlock::new(&model_uuid, None)));
        let mut source_block1 = OutputBlock::new(&model_uuid, parent_block.clone(), 0);
        let mut source_block2 = OutputBlock::new(&model_uuid, parent_block.clone(), 0);
        let target_block1_mutex = Arc::new(Mutex::new(OutputBlock::new(
            &model_uuid,
            parent_block.clone(),
            1,
        )));
        let target_block2_mutex = Arc::new(Mutex::new(OutputBlock::new(
            &model_uuid,
            parent_block.clone(),
            2,
        )));

        // create axon-section between the test-blocks
        {
            let mut target_block1 = target_block1_mutex.lock().expect("mutex poisoned");
            let mut target_block2 = target_block2_mutex.lock().expect("mutex poisoned");

            // test get of target position for target 1
            let target1_pos1 = target_block1.get_free_input();
            assert_eq!(target1_pos1, 0);
            let fail1_pos = target_block1.get_free_input();
            assert_eq!(fail1_pos, 255);
            // test get of target position for target 2
            let target2_pos1 = target_block2.get_free_input();
            assert_eq!(target2_pos1, 0);
            let target2_pos2 = target_block2.get_free_input();
            assert_eq!(target2_pos2, 1);
            let fail2_pos = target_block2.get_free_input();
            assert_eq!(fail2_pos, 255);

            // connect source 1 to target 1
            let mut source_axonsection1 = AxonSection::default();
            source_axonsection1.model_uuid = model_uuid.clone();
            source_axonsection1.target_block_uuid = target_block1.get_uuid();
            source_axonsection1.target_block = Some(target_block1_mutex.clone());
            source_axonsection1.target_pos = target1_pos1;
            source_block1
                .block_io
                .output_buffer
                .push(source_axonsection1);

            // connect source 1 to target 2
            let mut source_axonsection2 = AxonSection::default();
            source_axonsection2.model_uuid = model_uuid.clone();
            source_axonsection2.target_block_uuid = target_block2.get_uuid();
            source_axonsection2.target_block = Some(target_block2_mutex.clone());
            source_axonsection2.target_pos = target2_pos1;
            source_block1
                .block_io
                .output_buffer
                .push(source_axonsection2);

            // connect source 2 target 2
            let mut source_axonsection3 = AxonSection::default();
            source_axonsection3.model_uuid = model_uuid.clone();
            source_axonsection3.target_block_uuid = target_block2.get_uuid();
            source_axonsection3.target_block = Some(target_block2_mutex.clone());
            source_axonsection3.target_pos = target2_pos2;
            source_block2
                .block_io
                .output_buffer
                .push(source_axonsection3);
        }

        // +----------+    +----------+
        // | source 1 |--->| target 1 |
        // +----------+\   +----------+
        //              \
        //               \
        //                \
        //                 v
        // +----------+    +----------+
        // | source 2 |--->| target 2 |
        // +----------+    +----------+
        // ====================================== run tests =====================================
        // check target buffer at the beginning
        {
            assert_eq!(
                source_block1.block_io.output_buffer[0].is_already_send,
                false
            );
            assert_eq!(
                source_block1.block_io.output_buffer[1].is_already_send,
                false
            );
            assert_eq!(
                source_block2.block_io.output_buffer[0].is_already_send,
                false
            );

            let target_block1 = target_block1_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block1.block_io.input_buffer[0].is_ready_for_new_input,
                true
            );
            assert_eq!(target_block1.block_io.is_input_complete(), false);

            let target_block2 = target_block2_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block2.block_io.input_buffer[0].is_ready_for_new_input,
                true
            );
            assert_eq!(
                target_block2.block_io.input_buffer[1].is_ready_for_new_input,
                true
            );
            assert_eq!(target_block2.block_io.is_input_complete(), false);
        }
        // send from source 1 and check target buffer again
        assert_eq!(source_block1.process().unwrap(), true);
        {
            assert_eq!(
                source_block1.block_io.output_buffer[0].is_already_send,
                false
            );
            assert_eq!(
                source_block1.block_io.output_buffer[1].is_already_send,
                false
            );
            assert_eq!(
                source_block2.block_io.output_buffer[0].is_already_send,
                false
            );

            let target_block1 = target_block1_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block1.block_io.input_buffer[0].is_ready_for_new_input,
                false
            );
            assert_eq!(target_block1.block_io.is_input_complete(), true);

            let target_block2 = target_block2_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block2.block_io.input_buffer[0].is_ready_for_new_input,
                false
            );
            assert_eq!(
                target_block2.block_io.input_buffer[1].is_ready_for_new_input,
                true
            );
            assert_eq!(target_block2.block_io.is_input_complete(), false);
        }
        // test that source 1 has to be requeued, if target not ready for new input
        assert_eq!(source_block1.process().unwrap(), false);
        {
            assert_eq!(
                source_block1.block_io.output_buffer[0].is_already_send,
                false
            );
            assert_eq!(
                source_block1.block_io.output_buffer[1].is_already_send,
                false
            );
            assert_eq!(
                source_block2.block_io.output_buffer[0].is_already_send,
                false
            );

            let target_block1 = target_block1_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block1.block_io.input_buffer[0].is_ready_for_new_input,
                false
            );
            assert_eq!(target_block1.block_io.is_input_complete(), true);

            let target_block2 = target_block2_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block2.block_io.input_buffer[0].is_ready_for_new_input,
                false
            );
            assert_eq!(
                target_block2.block_io.input_buffer[1].is_ready_for_new_input,
                true
            );
            assert_eq!(target_block2.block_io.is_input_complete(), false);
        }

        // send from source 2 and check target buffer again
        assert_eq!(source_block2.process().unwrap(), true);
        {
            assert_eq!(
                source_block1.block_io.output_buffer[0].is_already_send,
                false
            );
            assert_eq!(
                source_block1.block_io.output_buffer[1].is_already_send,
                false
            );
            assert_eq!(
                source_block2.block_io.output_buffer[0].is_already_send,
                false
            );

            let target_block1 = target_block1_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block1.block_io.input_buffer[0].is_ready_for_new_input,
                false
            );
            assert_eq!(target_block1.block_io.is_input_complete(), true);

            let target_block2 = target_block2_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block2.block_io.input_buffer[0].is_ready_for_new_input,
                false
            );
            assert_eq!(
                target_block2.block_io.input_buffer[1].is_ready_for_new_input,
                false
            );
            assert_eq!(target_block2.block_io.is_input_complete(), true);
        }
        // process target 1 and check target buffer again
        {
            let mut target_block1 = target_block1_mutex.lock().expect("mutex poisoned");
            assert_eq!(target_block1.process().unwrap(), true);
            assert_eq!(
                target_block1.block_io.input_buffer[0].is_ready_for_new_input,
                true
            );
            assert_eq!(target_block1.block_io.is_input_complete(), false);

            let mut target_block2 = target_block2_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block2.block_io.input_buffer[0].is_ready_for_new_input,
                false
            );
            assert_eq!(
                target_block2.block_io.input_buffer[1].is_ready_for_new_input,
                false
            );
            assert_eq!(target_block2.block_io.is_input_complete(), true);
        }
        // process target 2 and check target buffer again
        {
            let mut target_block2 = target_block2_mutex.lock().expect("mutex poisoned");
            assert_eq!(target_block2.process().unwrap(), true);
            assert_eq!(
                target_block2.block_io.input_buffer[0].is_ready_for_new_input,
                true
            );
            assert_eq!(
                target_block2.block_io.input_buffer[1].is_ready_for_new_input,
                true
            );
            assert_eq!(target_block2.block_io.is_input_complete(), false);

            let mut target_block1 = target_block1_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block1.block_io.input_buffer[0].is_ready_for_new_input,
                true
            );
            assert_eq!(target_block1.block_io.is_input_complete(), false);
        }
    }

    #[test]
    fn test_send_forward_with_2_sources() {
        // create blocks for testing
        let model_uuid = Uuid::new_v4();
        let parent_block = Arc::new(Mutex::new(HexagonBlock::new(&model_uuid, None)));
        let mut source_block1 = OutputBlock::new(&model_uuid, parent_block.clone(), 0);
        let mut source_block2 = OutputBlock::new(&model_uuid, parent_block.clone(), 0);
        let target_block_mutex = Arc::new(Mutex::new(OutputBlock::new(
            &model_uuid,
            parent_block.clone(),
            2,
        )));

        // create axon-section between the test-blocks
        {
            let mut target_block = target_block_mutex.lock().expect("mutex poisoned");

            // test get of target position
            let target_pos1 = target_block.get_free_input();
            assert_eq!(target_pos1, 0);
            let target_pos2 = target_block.get_free_input();
            assert_eq!(target_pos2, 1);
            let fail_pos = target_block.get_free_input();
            assert_eq!(fail_pos, 255);

            // connect source 1
            let mut source_axonsection1 = AxonSection::default();
            source_axonsection1.model_uuid = model_uuid.clone();
            source_axonsection1.target_block_uuid = target_block.get_uuid();
            source_axonsection1.target_block = Some(target_block_mutex.clone());
            source_axonsection1.target_pos = target_pos1;
            source_block1
                .block_io
                .output_buffer
                .push(source_axonsection1);

            // connect source 2
            let mut source_axonsection2 = AxonSection::default();
            source_axonsection2.model_uuid = model_uuid.clone();
            source_axonsection2.target_block_uuid = target_block.get_uuid();
            source_axonsection2.target_block = Some(target_block_mutex.clone());
            source_axonsection2.target_pos = target_pos2;
            source_block2
                .block_io
                .output_buffer
                .push(source_axonsection2);
        }

        // +----------+
        // | source 1 |
        // +----------+\
        //              \
        //               \
        //                \
        //                 v
        // +----------+    +----------+
        // | source 2 |--->| target 2 |
        // +----------+    +----------+
        // ====================================== run tests =====================================
        // check target buffer at the beginning
        {
            let target_block = target_block_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block.block_io.input_buffer[0].is_ready_for_new_input,
                true
            );
            assert_eq!(
                target_block.block_io.input_buffer[1].is_ready_for_new_input,
                true
            );
            assert_eq!(target_block.block_io.is_input_complete(), false);
        }
        // send from first source and check target buffer again
        assert_eq!(source_block1.process().unwrap(), true);
        {
            let target_block = target_block_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block.block_io.input_buffer[0].is_ready_for_new_input,
                false
            );
            assert_eq!(
                target_block.block_io.input_buffer[1].is_ready_for_new_input,
                true
            );
            assert_eq!(target_block.block_io.is_input_complete(), false);
        }
        // send from second source and check target buffer again
        assert_eq!(source_block2.process().unwrap(), true);
        {
            let target_block = target_block_mutex.lock().expect("mutex poisoned");
            assert_eq!(
                target_block.block_io.input_buffer[0].is_ready_for_new_input,
                false
            );
            assert_eq!(
                target_block.block_io.input_buffer[1].is_ready_for_new_input,
                false
            );
            assert_eq!(target_block.block_io.is_input_complete(), true);
        }
        // process target and check target buffer again
        {
            let mut target_block = target_block_mutex.lock().expect("mutex poisoned");
            assert_eq!(target_block.process().unwrap(), true);
            assert_eq!(
                target_block.block_io.input_buffer[0].is_ready_for_new_input,
                true
            );
            assert_eq!(
                target_block.block_io.input_buffer[1].is_ready_for_new_input,
                true
            );
            assert_eq!(target_block.block_io.is_input_complete(), false);
        }
    }

    #[test]
    fn test_axonsection_serialize_deserialize() {
        let mut original = AxonSection {
            data: AxonData::default(),
            model_uuid: Uuid::new_v4(),
            task_id: 42,
            cycle_number: 42,
            do_train: true,
            target_block_uuid: Uuid::new_v4(),
            target_pos: 42,
            target_block: None,
            target_type: ObjectType::CoreBlock,
            is_already_send: false,
            is_ready_for_new_input: true,
        };

        // Modify specific axon values for testing
        original.data.axons[42] = 123.0f32;

        // Serialize the AxonSection
        let cfg = bincode::config::standard();
        let serialized: Vec<u8> =
            bincode::serde::encode_to_vec(&original, cfg).expect("Failed to serialize");
        let deserialized: AxonSection = bincode::serde::decode_from_slice(&serialized, cfg)
            .expect("Failed to deserialize")
            .0;

        // Print the size of the serialized data
        println!("size: {}", serialized.len());

        // Verify that the deserialized data matches the original
        assert_eq!(original, deserialized);
    }
}
