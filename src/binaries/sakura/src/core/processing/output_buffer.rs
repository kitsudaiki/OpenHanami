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
use std::cmp::min;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use ainari_common::enums::*;

use crate::core::processing::worker_queue::*;

use super::super::blocks::block_trait::*;
use super::super::blocks::output_block::*;

/// A buffer structure for storing output data from neural network processing.
/// This structure holds the output neurons, their types, and various metadata
/// needed for processing and backpropagation.
#[derive(Debug, Serialize, Deserialize)]
pub struct OutputBuffer {
    /// Unique identifier for this output buffer
    #[allow(dead_code)]
    pub uuid: Uuid,
    /// UUID of the hexagon this buffer belongs to
    #[allow(dead_code)]
    pub hexagon_uuid: Uuid,
    /// UUID of the model this buffer belongs to
    pub model_uuid: Uuid,
    /// Name of this output buffer
    pub name: String,

    /// Collection of output neurons containing the processed data
    pub output_neurons: Vec<OutputNeuron>,
    /// Type of output data this buffer holds
    pub output_type: OutputType,
    /// Size of the output data in bytes
    pub output_size: u64,

    /// Flag indicating whether this buffer has been finalized
    pub already_finalized: bool,
    /// Number of blocks connected to this output buffer
    pub number_of_connected_blocks: u64,
    /// List of blocks that haven't completed processing
    #[serde(skip, default = "init_unfinished_blocks")]
    pub unfinished_blocks: Vec<Arc<Mutex<dyn Block>>>,
}

impl PartialEq for OutputBuffer {
    /// Compares two OutputBuffers for equality based on their fields.
    /// This is used to determine if two buffers represent the same logical output.
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
            && self.hexagon_uuid == other.hexagon_uuid
            && self.model_uuid == other.model_uuid
            && self.name == other.name
            && self.output_neurons == other.output_neurons
            && self.output_type == other.output_type
            && self.output_size == other.output_size
            && self.already_finalized == other.already_finalized
            && self.number_of_connected_blocks == other.number_of_connected_blocks
    }
}

/// Initializes an empty vector for tracking unfinished blocks
/// This vector will hold blocks that haven't completed processing yet.
fn init_unfinished_blocks() -> Vec<Arc<Mutex<dyn Block>>> {
    Vec::new()
}

impl OutputBuffer {
    /// Creates a new OutputBuffer with the given parameters.
    /// This initializes all fields to their default values and sets up the basic structure.
    pub fn new(
        name: &str,
        hexagon_uuid: &Uuid,
        model_uuid: &Uuid,
        output_type: &OutputType,
    ) -> Self {
        OutputBuffer {
            uuid: *hexagon_uuid,
            hexagon_uuid: *hexagon_uuid,
            model_uuid: *model_uuid,
            name: name.to_owned(),

            output_neurons: Vec::new(),
            output_type: output_type.clone(),
            output_size: 0,

            already_finalized: false,
            number_of_connected_blocks: 0,
            unfinished_blocks: Vec::new(),
        }
    }

    /// Updates the buffer size and allocates space for the specified number of outputs.
    /// This resizes the output_neurons vector to accommodate the new size and adjusts
    /// the size based on the output type (float or int outputs require more space).
    pub fn update_buffer(&mut self, number_of_outputs: usize) {
        let mut number_of_outputs_copy = number_of_outputs;

        if self.output_size < number_of_outputs_copy as u64 {
            self.output_size = number_of_outputs_copy as u64;

            // For float outputs, each output is represented by 32 bits (1 neuron per bit)
            if self.output_type == OutputType::FloatOutput {
                number_of_outputs_copy *= 32;
            }
            // For int outputs, each output is represented by 64 bits (1 neuron per bit)
            if self.output_type == OutputType::IntOutput {
                number_of_outputs_copy *= 64;
            }

            // Resize the output neurons vector, initializing new elements with default values
            self.output_neurons
                .resize_with(number_of_outputs_copy, OutputNeuron::default);
        }
    }

    /// Finalizes the training process by applying the sigmoid activation function
    /// to all output neurons. This transforms the raw output values into probabilities.
    pub fn finalize_train(&mut self) {
        for out in self.output_neurons.iter_mut() {
            if out.output_value != 0.0f32 {
                // Apply sigmoid function: 1 / (1 + e^(-x))
                out.output_value = 1.0f32 / (1.0f32 + (-out.output_value).exp());
            }
        }

        self.already_finalized = true;
    }

    /// Finalizes the processing by applying the sigmoid activation function
    /// and clearing the list of unfinished blocks.
    pub fn finalize_processing(&mut self) {
        for out in self.output_neurons.iter_mut() {
            if out.output_value != 0.0f32 {
                // Apply sigmoid function: 1 / (1 + e^(-x))
                out.output_value = 1.0f32 / (1.0f32 + (-out.output_value).exp());
            }
        }

        self.already_finalized = true;
        self.unfinished_blocks.clear();
    }

    /// Performs backpropagation by calculating the error for each output neuron
    /// and scheduling backpropagation tasks for connected blocks.
    pub fn backpropagate(&mut self, cycle_number: u64) {
        // Calculate the error for each output neuron
        for out in self.output_neurons.iter_mut() {
            let delta = out.output_value - out.expected_value;
            // Calculate the gradient for backpropagation
            out.expected_value = delta * out.output_value * (1.0f32 - out.output_value);
        }

        // Get the worker queue to schedule backpropagation tasks
        let mut worker_queue = WORKER_QUEUE.lock().expect("mutex poisoned");
        for block in self.unfinished_blocks.iter() {
            // let worker_task = WorkerTask {
            //     task_type: WorkerTaskType::Backpropagate,
            //     block: Arc::clone(block),
            //     cycle_number,
            // };

            // Add the task to the worker queue
            // worker_queue.add(worker_task);
        }
        self.unfinished_blocks.clear();
    }

    /// Resets the output values of all neurons to 0 and resets the local finish counter.
    pub fn reset_output(&mut self) {
        for out in self.output_neurons.iter_mut() {
            out.output_value = 0.0f32;
        }
    }

    /// Serializes the OutputBuffer to a byte vector using bincode.
    /// This allows the buffer to be stored or transmitted efficiently.
    pub fn serailize(&self) -> Vec<u8> {
        let cfg = bincode::config::standard();
        bincode::serde::encode_to_vec(self, cfg).expect("Failed to serialize")
    }

    /// Updates the finish counter and checks if all connected blocks have completed processing.
    /// Returns true if the buffer is ready for the next processing cycle.
    pub fn update_finish_counter(&mut self, cycle_number: u64) -> bool {
        // let finish_counter = self.finish_counter_mutex.lock().expect("mutex poisoned");
        // let expected_cycle_number = finish_counter.get_expected_cycle_number();
        // if cycle_number == expected_cycle_number {
        //     self.local_finish_counter += 1;
        //     if self.local_finish_counter >= self.number_of_connected_blocks {
        //         return true;
        //     }
        // }

        false
    }
}
