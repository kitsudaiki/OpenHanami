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

use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use ainari_common::constants::*;
use ainari_common::enums::*;
use ainari_common::error::AinariError;

use super::hexagon_block::*;
use super::*;

// ==================================================================================================

/// Represents a neuron in the output layer of a neural network block.
/// Contains both the computed output value and the expected value for training purposes.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct OutputNeuron {
    /// The computed output value of this neuron
    pub output_value: f32,
    /// The expected output value used for training
    pub expected_value: f32,
}

impl OutputNeuron {
    /// Creates a new OutputNeuron with default values (0.0 for both fields)
    pub fn default() -> Self {
        OutputNeuron {
            output_value: 0.0f32,
            expected_value: 0.0f32,
        }
    }
}

// ==================================================================================================

/// Represents an output block in the neural network that collects and processes outputs.
/// This block connects to an output buffer to aggregate results from multiple blocks.
#[derive(Debug, Serialize)]
pub struct OutputBlock {
    pub uuid: Uuid,
    pub model_uuid: Uuid,
    #[serde(skip)]
    pub parent_block: Arc<Mutex<HexagonBlock>>,

    pub is_processed: bool,

    pub block_io: BlockIoBuffer,

    pub weights: Vec<f32>,
    pub block_outputs: Vec<OutputNeuron>,

    pub was_already_connected: bool,
}

impl PartialEq for OutputBlock {
    /// Compares two OutputBlocks for equality by comparing all their fields
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
            && self.model_uuid == other.model_uuid
            && self.block_io == other.block_io
            && self.weights == other.weights
            && self.block_outputs == other.block_outputs
            && self.was_already_connected == other.was_already_connected
    }
}

impl OutputBlock {
    /// Creates a new OutputBlock with default values
    ///
    /// # Arguments
    ///
    /// * `hexagon_uuid` - UUID of the hexagon this block belongs to
    /// * `model_uuid` - UUID of the model this block belongs to
    /// * `output_buffer_name` - Name of the output buffer this block should connect to
    pub fn new(
        model_uuid: &Uuid,
        parent_block: Arc<Mutex<HexagonBlock>>,
        number_of_inputs: usize,
    ) -> Self {
        OutputBlock {
            uuid: Uuid::new_v4(),
            model_uuid: *model_uuid,
            parent_block: parent_block,

            is_processed: false,

            block_io: BlockIoBuffer::new(number_of_inputs),

            weights: Vec::new(),
            block_outputs: Vec::new(),

            was_already_connected: false,
        }
    }

    /// Connects this block to its output buffer if not already connected
    ///
    /// # Returns
    ///
    /// * `Ok(())` if connection was successful or already established
    /// * `Err(AinariError)` if connection failed
    fn connect_output_buffer(&mut self) -> Result<(), AinariError> {
        // // connect output-buffer if not already done
        // if self.output_buffer.is_none() {
        //     let root_handler = MODEL_HANDLER.read().expect("mutex poisoned");
        //     let output_buffer_mutex =
        //         root_handler.get_output_buffer(&self.model_uuid, &self.output_buffer_name)?;

        //     self.output_buffer = Some(output_buffer_mutex.clone());
        //     let mut output_buffer = output_buffer_mutex.lock().expect("mutex poisoned");
        //     // after a checkpoint-restore the block must be connected to the buffer again,
        //     // but is not allowed to increase the counter further
        //     if !self.was_already_connected {
        //         output_buffer.number_of_connected_blocks += 1;
        //     }
        //     self.was_already_connected = true;
        // }

        Ok(())
    }

    /// Processes the input to produce output values
    ///
    /// Resets all output values, calculates the activation of each input axon,
    /// and computes the weighted sum for each output neuron.
    fn process_block(&mut self) {
        // // reset output-values
        // for output_neuron in self.block_outputs.iter_mut() {
        //     output_neuron.output_value = 0.0f32;
        // }

        // let input_buffer = &mut self.block_io.input_buffer[0];
        // // calculate block-internal output
        // for (x, axon) in input_buffer.data.axons.iter_mut().enumerate() {
        //     if axon.potential == 0.0f32 {
        //         continue;
        //     }

        //     // Apply sigmoid activation function
        //     axon.potential = 1.0f32 / (1.0f32 + (-axon.potential).exp());
        //     for (y, output_neuron) in self.block_outputs.iter_mut().enumerate() {
        //         // Calculate weighted sum for each output neuron
        //         output_neuron.output_value += self.weights[(y * BLOCK_DIM) + x] * axon.potential;
        //     }
        // }
    }

    // /// Finalizes the training process by applying the sigmoid activation function
    // /// to all output neurons. This transforms the raw output values into probabilities.
    // pub fn finalize_train(&mut self) {
    //     for out in self.output_neurons.iter_mut() {
    //         if out.output_value != 0.0f32 {
    //             // Apply sigmoid function: 1 / (1 + e^(-x))
    //             out.output_value = 1.0f32 / (1.0f32 + (-out.output_value).exp());
    //         }
    //     }

    //     self.already_finalized = true;
    // }

    // /// Finalizes the processing by applying the sigmoid activation function
    // /// and clearing the list of unfinished blocks.
    // pub fn finalize_processing(&mut self) {
    //     for out in self.output_neurons.iter_mut() {
    //         if out.output_value != 0.0f32 {
    //             // Apply sigmoid function: 1 / (1 + e^(-x))
    //             out.output_value = 1.0f32 / (1.0f32 + (-out.output_value).exp());
    //         }
    //     }

    //     self.already_finalized = true;
    //     self.unfinished_blocks.clear();
    // }

    // /// Performs backpropagation by calculating the error for each output neuron
    // /// and scheduling backpropagation tasks for connected blocks.
    // pub fn backpropagate(&mut self, cycle_number: u64) {
    //     // Calculate the error for each output neuron
    //     for out in self.output_neurons.iter_mut() {
    //         let delta = out.output_value - out.expected_value;
    //         // Calculate the gradient for backpropagation
    //         out.expected_value = delta * out.output_value * (1.0f32 - out.output_value);
    //     }

    //     // Get the worker queue to schedule backpropagation tasks
    //     let mut worker_queue = WORKER_QUEUE.lock().expect("mutex poisoned");
    //     for block in self.unfinished_blocks.iter() {
    //         // let worker_task = WorkerTask {
    //         //     task_type: WorkerTaskType::Backpropagate,
    //         //     block: Arc::clone(block),
    //         //     cycle_number,
    //         // };

    //         // Add the task to the worker queue
    //         // worker_queue.add(worker_task);
    //     }
    //     self.unfinished_blocks.clear();
    // }
}

// ==================================================================================================

impl Block for OutputBlock {
    /// Trains the block by adjusting its weights based on the error between expected and actual outputs
    ///
    /// # Arguments
    ///
    /// * `_` - Unused parameter (reserved for future use)
    /// * `own` - Arc<Mutex<dyn Block>> reference to this block
    /// * `cycle_number` - The current training cycle number
    ///
    /// # Returns
    ///
    /// * `Ok(Some(finish_counter))` if training is complete and a finish counter is needed
    /// * `Ok(None)` if training is not yet complete
    /// * `Err(AinariError)` if an error occurs during training
    fn process(&mut self) -> Result<bool, AinariError> {
        // self.connect_output_buffer()?;

        // // resize output and wights and get expected values from output-buffer
        // if let Some(output_buffer_mutex) = &self.output_buffer {
        //     let mut rng = rand::rng();

        //     let output_buffer = output_buffer_mutex.lock().expect("mutex poisoned");
        //     self.block_outputs
        //         .resize_with(output_buffer.output_neurons.len(), OutputNeuron::default);
        //     let number_fo_weights = self.block_outputs.len() * BLOCK_DIM;
        //     self.weights
        //         .resize_with(number_fo_weights, || rng.random_range(-0.5..0.5));
        // } else {
        //     // TODO: error handling
        // }

        // self.process_block();

        // // process output-buffer
        // let mut already_done = false;
        // if let Some(output_buffer_mutex) = &self.output_buffer {
        //     let mut output_buffer = output_buffer_mutex.lock().expect("mutex poisoned");
        //     for (i, local_neuron) in self.block_outputs.iter().enumerate() {
        //         output_buffer.output_neurons[i].output_value += local_neuron.output_value;
        //     }

        //     if !output_buffer.already_finalized {
        //         if output_buffer.update_finish_counter(cycle_number) {
        //             output_buffer.finalize_train();
        //             output_buffer.backpropagate(cycle_number);
        //             already_done = true;
        //         } else {
        //             //output_buffer.unfinished_blocks.push(own);
        //         }
        //     } else {
        //         already_done = true;
        //     }
        // }
        if !self.is_processed {}

        self.is_processed = true;

        let is_finished = self.block_io.send_forward()?;
        if is_finished {
            self.is_processed = false;
        }
        Ok(is_finished)
    }

    // /// Processes the block without training, simply computing outputs
    // ///
    // /// # Arguments
    // ///
    // /// * `cycle_number` - The current processing cycle number
    // ///
    // /// # Returns
    // ///
    // /// * `Ok(Some(finish_counter))` if processing is complete and a finish counter is needed
    // /// * `Ok(None)` if processing is not yet complete
    // /// * `Err(AinariError)` if an error occurs during processing
    // fn process(
    //     &mut self,
    //     cycle_number: u64,
    // ) -> Result<Option<Arc<Mutex<FinishCounter>>>, AinariError> {
    //     self.connect_output_buffer()?;
    //     self.process_block();

    //     let mut finish_counter_option = None;

    //     // process output-buffer
    //     if let Some(output_buffer_mutex) = &self.output_buffer {
    //         let mut output_buffer = output_buffer_mutex.lock().expect("mutex poisoned");
    //         for (i, local_neuron) in self.block_outputs.iter().enumerate() {
    //             output_buffer.output_neurons[i].output_value += local_neuron.output_value;
    //         }

    //         if output_buffer.update_finish_counter(cycle_number) {
    //             output_buffer.finalize_processing();
    //             finish_counter_option = Some(output_buffer.finish_counter_mutex.clone());
    //         }
    //     }

    //     Ok(finish_counter_option)
    // }

    // /// Performs backpropagation to adjust weights based on errors
    // ///
    // /// # Arguments
    // ///
    // /// * `cycle_number` - The current backpropagation cycle number
    // ///
    // /// # Returns
    // ///
    // /// * `Ok(None)` if backpropagation was successful
    // /// * `Err(AinariError)` if an error occurs during backpropagation
    // fn backpropagate(
    //     &mut self,
    //     cycle_number: u64,
    // ) -> Result<Option<Arc<Mutex<FinishCounter>>>, AinariError> {
    //     self.connect_output_buffer()?;

    //     // resize output and wights and get expected values from output-buffer
    //     if let Some(output_buffer_mutex) = &self.output_buffer {
    //         let output_buffer = output_buffer_mutex.lock().expect("mutex poisoned");
    //         self.block_outputs
    //             .resize_with(output_buffer.output_neurons.len(), OutputNeuron::default);
    //         for i in 0..self.block_outputs.len() {
    //             self.block_outputs[i].expected_value =
    //                 output_buffer.output_neurons[i].expected_value;
    //         }
    //     } else {
    //         // TODO: error
    //     }

    //     // backpropagate block
    //     let input_buffer = &mut self.block_io.input_buffer[0];
    //     for (x, axon) in input_buffer.data.axons.iter_mut().enumerate() {
    //         axon.delta = 0.0f32;
    //         if axon.potential == 0.0f32 {
    //             continue;
    //         }

    //         for (y, output_neuron) in self.block_outputs.iter_mut().enumerate() {
    //             let weight = &mut self.weights[(y * BLOCK_DIM) + x];
    //             let update = output_neuron.expected_value;
    //             axon.delta += update * (*weight);
    //             *weight -= update * OUTPUT_TRAIN_VALUE * axon.potential;
    //         }

    //         axon.delta *= axon.potential * (1.0f32 - axon.potential);
    //     }

    //     send_backward(&mut self.block_io, cycle_number);

    //     Ok(None)
    // }

    /// Gets a free input slot in the block's input buffer.
    ///
    /// This function allocates an available input slot for an axon section.
    /// It manages the input buffer and tracks used slots.
    ///
    /// # Arguments
    ///
    /// * `axon_section` - The axon section to be assigned to a free input slot
    ///
    /// # Returns
    ///
    /// * `true` if an input slot was successfully allocated
    /// * `false` if no input slots are available
    fn get_free_input(&mut self) -> u8 {
        self.block_io.get_free_input()
    }

    // /// Finalizes the training process for this block
    // ///
    // /// # Arguments
    // ///
    // /// * `_` - Unused parameter (reserved for future use)
    // ///
    // /// # Returns
    // ///
    // /// * `Ok(())` if finalization was successful
    // /// * `Err(AinariError)` if an error occurs during finalization
    // fn finalize_train(&mut self, _: u64) -> Result<(), AinariError> {
    //     Ok(())
    // }

    // /// Finalizes the processing of this block
    // ///
    // /// # Arguments
    // ///
    // /// * `_` - Unused parameter (reserved for future use)
    // ///
    // /// # Returns
    // ///
    // /// * `Ok(())` if finalization was successful
    // /// * `Err(AinariError)` if an error occurs during finalization
    // fn finalize_process(&mut self, _: u64) -> Result<(), AinariError> {
    //     Ok(())
    // }

    // /// Finalizes the backpropagation process for this block
    // ///
    // /// # Arguments
    // ///
    // /// * `_` - Unused parameter (reserved for future use)
    // ///
    // /// # Returns
    // ///
    // /// * `Ok(true)` if finalization was successful
    // /// * `Ok(false)` if finalization was not needed
    // /// * `Err(AinariError)` if an error occurs during finalization
    // fn finalize_backpropagate(&mut self, _: u64) -> Result<bool, AinariError> {
    //     Ok(true)
    // }

    /// Gets the UUID of this block
    ///
    /// # Returns
    ///
    /// The UUID of this block
    fn get_uuid(&self) -> Uuid {
        self.uuid
    }

    /// Gets the UUID of the model this block belongs to
    ///
    /// # Returns
    ///
    /// The UUID of the model
    fn get_model_uuid(&self) -> Uuid {
        self.model_uuid
    }

    /// Gets a mutable reference to the block's I/O buffer
    ///
    /// # Returns
    ///
    /// A mutable reference to the BlockIoBuffer
    fn get_block_io(&mut self) -> &mut BlockIoBuffer {
        &mut self.block_io
    }

    /// Gets the type of this block
    ///
    /// # Returns
    ///
    /// The ObjectType of this block (always OutputBlock)
    fn get_type(&self) -> ObjectType {
        ObjectType::OutputBlock
    }

    fn get_parent_block(&self) -> Option<Arc<Mutex<HexagonBlock>>> {
        Some(self.parent_block.clone())
    }

    /// Serializes this block to a byte vector
    ///
    /// # Returns
    ///
    /// A byte vector containing the serialized block
    fn serailize(&self) -> Vec<u8> {
        let cfg = bincode::config::standard();
        bincode::serde::encode_to_vec(self, cfg).expect("Failed to serialize")
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_serialize_deserialize() {
//         let original = OutputBlock::new(&Uuid::new_v4(), &Uuid::new_v4(), 1);

//         let cfg = bincode::config::standard();
//         let serialized: Vec<u8> =
//             bincode::serde::encode_to_vec(&original, cfg).expect("Failed to serialize");
//         let deserialized: OutputBlock = bincode::serde::decode_from_slice(&serialized, cfg)
//             .expect("Failed to deserialize")
//             .0;
//         println!("size: {}", serialized.len());

//         assert_eq!(original, deserialized);
//     }
// }
