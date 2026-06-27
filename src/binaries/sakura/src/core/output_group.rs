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
use std::cmp::min;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use uuid::Uuid;

use ainari_common::enums::*;
use ainari_common::error::AinariError;
use ainari_dataset::dataset_io::{DataSetFileReadHandle, DataSetFileWriteHandle};
use ainari_model_parser::model_meta_structs::*;

use crate::core::blocks::core_block::*;
use crate::core::blocks::input_block::*;
use crate::core::blocks::output_block::*;
use crate::core::blocks::start_end_block::*;
use crate::core::blocks::transfer_block::*;

use super::blocks::block_trait::Block;

pub struct OutputGroup {
    pub output_type: OutputType,
    pub output_size: u64,
    pub output_values: Vec<f32>,
    pub expected_values: Vec<f32>,
    pub blocks: Vec<Arc<Mutex<OutputBlock>>>,
}

impl OutputGroup {
    pub fn new(output_type: &OutputType) -> Self {
        OutputGroup {
            output_type: output_type.clone(),
            output_size: 0,
            output_values: Vec::new(),
            expected_values: Vec::new(),
            blocks: Vec::new(),
        }
    }

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
            self.output_values.resize(number_of_outputs_copy, 0.0f32);
            self.expected_values.resize(number_of_outputs_copy, 0.0f32);
        }
    }

    pub fn apply_expected(
        &mut self, 
        input_ptr: &[f32],
        input_size: u64,
    ) -> Result<(), AinariError> {

        convert_buffer_to_expected(self, input_ptr, input_size);
        // TODO: handle result
        Ok(())
    }

    fn apply_dataset_to_expected(
        &mut self, 
        file_handle: &mut DataSetFileReadHandle,
        cycle_count: u64,
        time_length: u64,
        forecast_length: u64,
    ) -> Result<(), AinariError> {

        if forecast_length == 0 {
            let (input_ptr, input_size) =
                file_handle.get_data_from_file(&(cycle_count + time_length - 1))?;
            convert_buffer_to_expected(self, input_ptr, input_size);
        } else {
            // fill input with data from dataset
            let (_, row_size) = file_handle.get_data_from_file(&(cycle_count + time_length))?;
            let mut input_buffer = vec![0.0f32; (row_size * forecast_length) as usize];
            for cycle_internal_time_point in 0..forecast_length {
                let row_number =
                    (cycle_count * forecast_length) + time_length + cycle_internal_time_point;
                let (input_ptr, input_size) = file_handle.get_data_from_file(&row_number)?;
                let start = (cycle_internal_time_point * row_size) as usize;
                input_buffer[start..start + input_size as usize].copy_from_slice(input_ptr);
            }

            convert_buffer_to_expected(self, &input_buffer, input_buffer.len() as u64);
        }

        Ok(())
    }
}

/// Converts the output buffer's data to a flat buffer of f32 values.
/// The conversion depends on the output type (plain, bool, int, or float).
/// Returns the number of elements written to the buffer.
pub fn convert_output_to_buffer(buffer: &mut Vec<f32>, output_group: &OutputGroup) -> usize {
    match output_group.output_type {
        OutputType::PlainOutput => handle_plain_output(buffer, output_group),
        OutputType::BoolOutput => handle_bool_output(buffer, output_group),
        OutputType::IntOutput => handle_int_output(buffer, output_group),
        OutputType::FloatOutput => handle_float_output(buffer, output_group),
    }
}

/// Converts a flat buffer of f32 values to expected values in the output buffer.
/// The conversion depends on the output type (plain, bool, int, or float).
/// Returns the number of elements read from the buffer.
pub fn convert_buffer_to_expected(
    output_group: &mut OutputGroup,
    buffer: &[f32],
    buffer_size: u64,
) -> u64 {
    output_group.update_buffer(buffer.len());
    match output_group.output_type {
        OutputType::PlainOutput => handle_plain_expected(output_group, buffer, buffer_size),
        OutputType::BoolOutput => handle_bool_expected(output_group, buffer, buffer_size),
        OutputType::IntOutput => handle_int_expected(output_group, buffer, buffer_size),
        OutputType::FloatOutput => handle_float_expected(output_group, buffer, buffer_size),
    }
}

/// Handles conversion of plain output type to a flat buffer.
/// Copies the output values directly to the buffer.
fn handle_plain_output(buffer: &mut Vec<f32>, output_group: &OutputGroup) -> usize {
    buffer.resize(output_group.output_values.len(), 0.0f32);

    let number_of_outputs = min(buffer.len(), output_group.output_values.len());

    for (i, buffer) in buffer.iter_mut().enumerate().take(number_of_outputs) {
        *buffer = output_group.output_values[i];
    }

    number_of_outputs
}

/// Handles conversion of bool output type to a flat buffer.
/// Converts output values to 0.0 or 1.0 based on a threshold of 0.5.
fn handle_bool_output(buffer: &mut Vec<f32>, output_group: &OutputGroup) -> usize {
    buffer.resize(output_group.output_values.len(), 0.0f32);

    let number_of_outputs = min(buffer.len(), output_group.output_values.len());

    for (i, buffer) in buffer.iter_mut().enumerate().take(number_of_outputs) {
        *buffer = (output_group.output_values[i] >= 0.5f32) as u8 as f32;
    }

    number_of_outputs
}

/// Handles conversion of int output type to a flat buffer.
/// Combines 64 neurons into a single integer value.
fn handle_int_output(buffer: &mut Vec<f32>, output_group: &OutputGroup) -> usize {
    buffer.resize(output_group.output_values.len() / 64, 0.0f32);
    let number_of_outputs = min(buffer.len(), output_group.output_values.len() / 64);

    for (i, buffer) in buffer.iter_mut().enumerate().take(number_of_outputs) {
        let mut val: u64 = 0;

        for offset in 0..64 {
            let neuron = output_group.output_values[i * 64 + offset];
            val = (val << 1) | ((neuron >= 0.50) as u64);
        }

        *buffer = val as f32;
    }

    number_of_outputs
}

/// Handles conversion of float output type to a flat buffer.
/// Combines 32 neurons into a single float value using bit packing.
fn handle_float_output(buffer: &mut Vec<f32>, output_group: &OutputGroup) -> usize {
    buffer.resize(output_group.output_values.len() / 32, 0.0f32);
    let number_of_outputs = min(buffer.len(), output_group.output_values.len() / 32);

    for (i, buffer) in buffer.iter_mut().enumerate().take(number_of_outputs) {
        let mut val: u32 = 0;

        for offset in 0..32 {
            let neuron = output_group.output_values[i * 32 + offset];
            val = (val << 1) | ((neuron >= 0.5) as u32);
        }

        *buffer = f32::from_bits(val);
    }

    number_of_outputs
}

/// Handles setting expected values for plain output type.
/// Copies the values directly from the buffer to the expected values.
fn handle_plain_expected(output_group: &mut OutputGroup, buffer: &[f32], buffer_size: u64) -> u64 {
    let number_of_outputs = min(buffer_size, output_group.output_values.len() as u64);

    for i in 0..number_of_outputs {
        output_group.expected_values[i as usize] = buffer[i as usize];
    }

    number_of_outputs
}

/// Handles setting expected values for bool output type.
/// Converts values to 0.0 or 1.0 based on a threshold of 0.5.
fn handle_bool_expected(output_group: &mut OutputGroup, buffer: &[f32], buffer_size: u64) -> u64 {
    let number_of_outputs = min(buffer_size, output_group.output_values.len() as u64);

    for i in 0..number_of_outputs {
        output_group.expected_values[i as usize] = (buffer[i as usize] >= 0.5f32) as u8 as f32;
    }

    number_of_outputs
}

/// Handles setting expected values for int output type.
/// Expands a single integer value into 64 neurons.
fn handle_int_expected(output_group: &mut OutputGroup, buffer: &[f32], buffer_size: u64) -> u64 {
    let number_of_outputs = min(buffer_size, output_group.output_values.len() as u64 / 64);

    for i in 0..number_of_outputs {
        let val = buffer[i as usize] as u64;

        for offset in 0..64 {
            let index = (i * 64) + (63 - offset);
            output_group.expected_values[index as usize] = ((val >> offset) & 1) as u8 as f32;
        }
    }

    number_of_outputs
}

/// Handles setting expected values for float output type.
/// Expands a single float value into 32 neurons using bit unpacking.
fn handle_float_expected(output_group: &mut OutputGroup, buffer: &[f32], buffer_size: u64) -> u64 {
    let number_of_outputs = min(buffer_size, output_group.output_values.len() as u64 / 32);

    for i in 0..number_of_outputs {
        let val = buffer[i as usize].to_bits();

        for offset in 0..32 {
            let index = (i * 32) + (31 - offset);
            output_group.expected_values[index as usize] = ((val >> offset) & 1) as u8 as f32;
        }
    }

    number_of_outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain() {
        let mut output_group = OutputGroup::new(&OutputType::PlainOutput);
        output_group.update_buffer(4);

        let mut buffer: Vec<f32> = Vec::new();
        buffer.resize(4, 0.0f32);

        {
            output_group.output_values[0] = 42.0f32;
            output_group.output_values[1] = 43.0f32;
            output_group.output_values[2] = 44.0f32;
            output_group.output_values[3] = 45.0f32;
        }

        convert_output_to_buffer(&mut buffer, &mut output_group);

        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer[0], 42.0f32);
        assert_eq!(buffer[1], 43.0f32);
        assert_eq!(buffer[2], 44.0f32);
        assert_eq!(buffer[3], 45.0f32);

        convert_buffer_to_expected(&mut output_group, &buffer[..], buffer.len() as u64);

        assert_eq!(buffer.len(), 4);

        {
            assert_eq!(output_group.expected_values[0], 42.0f32);
            assert_eq!(output_group.expected_values[1], 43.0f32);
            assert_eq!(output_group.expected_values[2], 44.0f32);
            assert_eq!(output_group.expected_values[3], 45.0f32);
        }
    }

    #[test]
    fn test_bool() {
        let mut output_group = OutputGroup::new(&OutputType::BoolOutput);
        output_group.update_buffer(4);

        let mut buffer: Vec<f32> = Vec::new();
        buffer.resize(4, 0.0f32);

        {
            output_group.output_values[0] = 0.1f32;
            output_group.output_values[1] = 0.6f32;
            output_group.output_values[2] = 0.3f32;
            output_group.output_values[3] = 0.8f32;
        }

        convert_output_to_buffer(&mut buffer, &mut output_group);

        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer[0], 0.0f32);
        assert_eq!(buffer[1], 1.0f32);
        assert_eq!(buffer[2], 0.0f32);
        assert_eq!(buffer[3], 1.0f32);

        convert_buffer_to_expected(&mut output_group, &buffer[..], buffer.len() as u64);

        assert_eq!(buffer.len(), 4);

        {
            assert_eq!(output_group.expected_values[0], 0.0f32);
            assert_eq!(output_group.expected_values[1], 1.0f32);
            assert_eq!(output_group.expected_values[2], 0.0f32);
            assert_eq!(output_group.expected_values[3], 1.0f32);
        }
    }

    #[test]
    fn test_float() {
        let mut output_group = OutputGroup::new(&OutputType::FloatOutput);
        output_group.update_buffer(2);

        let mut buffer: Vec<f32> = Vec::new();
        buffer.resize(2, 0.0f32);

        {
            assert_eq!(output_group.output_values.len(), 64);
            output_group.output_values[15] = 0.6f32;
            output_group.output_values[16] = 0.1f32;
            output_group.output_values[42] = 0.3f32;
            output_group.output_values[43] = 0.8f32;
        }

        convert_output_to_buffer(&mut buffer, &mut output_group);

        assert_eq!(buffer.len(), 2);

        convert_buffer_to_expected(&mut output_group, &buffer[..], buffer.len() as u64);

        assert_eq!(buffer.len(), 2);

        {
            assert_eq!(output_group.expected_values[15], 1.0f32);
            assert_eq!(output_group.expected_values[16], 0.0f32);
            assert_eq!(output_group.expected_values[42], 0.0f32);
            assert_eq!(output_group.expected_values[43], 1.0f32);
        }
    }

    #[test]
    fn test_int() {
        let mut output_group = OutputGroup::new(&OutputType::IntOutput);
        output_group.update_buffer(2);

        let mut buffer: Vec<f32> = Vec::new();
        buffer.resize(2, 0.0f32);

        {
            assert_eq!(output_group.output_values.len(), 128);
            output_group.output_values[62] = 0.6f32;
            output_group.output_values[63] = 0.1f32;
            output_group.output_values[126] = 0.3f32;
            output_group.output_values[127] = 0.8f32;
        }

        convert_output_to_buffer(&mut buffer, &mut output_group);

        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer[0], 2.0f32);
        assert_eq!(buffer[1], 1.0f32);

        convert_buffer_to_expected(&mut output_group, &buffer[..], buffer.len() as u64);

        assert_eq!(buffer.len(), 2);
        {
            assert_eq!(output_group.expected_values[62], 1.0f32);
            assert_eq!(output_group.expected_values[63], 0.0f32);
            assert_eq!(output_group.expected_values[126], 0.0f32);
            assert_eq!(output_group.expected_values[127], 1.0f32);
        }
    }

    // #[test]
    // fn test_serialize_deserialize() {
    //     let original = OutputGroup::new(
    //         "test",
    //         &Uuid::new_v4(),
    //         &Uuid::new_v4(),
    //         &OutputType::PlainOutput,
    //     );

    //     let cfg = bincode::config::standard();
    //     let serialized: Vec<u8> =
    //         bincode::serde::encode_to_vec(&original, cfg).expect("Failed to serialize");
    //     let deserialized: OutputGroup = bincode::serde::decode_from_slice(&serialized, cfg)
    //         .expect("Failed to deserialize")
    //         .0;
    //     println!("size: {}", serialized.len());

    //     assert_eq!(original, deserialized);
    // }
}
