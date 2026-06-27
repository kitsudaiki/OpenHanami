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
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::sync::{Arc, Mutex, RwLock};
use uuid::Uuid;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use ainari_common::enums::*;
use ainari_common::error::AinariError;
use ainari_dataset::dataset_io::{DataSetFileReadHandle, DataSetFileWriteHandle};
use ainari_model_parser::model_meta_structs::*;

use crate::core::blocks::core_block::*;
use crate::core::blocks::input_block::*;
use crate::core::blocks::output_block::*;
use crate::core::blocks::start_end_block::*;
use crate::core::blocks::transfer_block::*;
use crate::core::input_group::*;
use crate::core::output_group::*;

use super::processing::task_queue::{TaskQueue, init_task_queue};
use super::processing::tasks::{self, Task, TaskVariant};
use super::blocks::block_trait::Block;

// ==================================================================================================

pub struct HexagonData {
    pub blocks: HashMap<Uuid, Arc<Mutex<dyn Block>>>,
}

impl HexagonData {
    pub fn new() -> Self {
        HexagonData {
            blocks: HashMap::new(),
        }
    }
}

// ==================================================================================================

pub struct ModelContent {
    /// Metadata describing the model's structure and configuration.
    pub model_meta: ModelMeta,

    pub hexagon_data: HashMap<Uuid, Arc<RwLock<HexagonData>>>,
    pub input_groups: HashMap<String, Arc<RwLock<InputGroup>>>,
    pub output_groups: HashMap<String, Arc<RwLock<OutputGroup>>>,

    pub transfer_block: Arc<Mutex<TransferBlock>>,
    pub start_end_block: Arc<Mutex<StartEndBlock>>,

    pub queue: Arc<Mutex<TaskQueue>>,
    pub handle: Option<JoinHandle<()>>,
    pub running: Arc<AtomicBool>,
}

// ==================================================================================================

/// Initializes a new model with the given metadata and UUID.
///
/// This creates a complete model structure including all blocks, inputs, and outputs.
///
/// # Arguments
/// * `model_uuid` - UUID of the model to initialize.
/// * `model_meta` - Metadata describing the model's structure and configuration.
///
/// # Returns
/// * `Ok(())` on success.
/// * `Err(AinariError)` if the model already exists or if initialization fails.
pub fn init_new_model(
    model_uuid: &Uuid,
    model_meta: ModelMeta,
) -> Result<ModelContent, AinariError> {
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);

    let queue = Arc::new(Mutex::new(init_task_queue()));
    let queue_clone = Arc::clone(&queue);

    let handle = thread::spawn(move || {
        log::debug!("Started model-thread");
        while running_clone.load(Ordering::Relaxed) {
            // get task from the task-queue and process the task, otherwise sleep until the next check
            let mut queue_handle = queue_clone.lock().expect("mutex poisoned");
            if let Some(task_mutex) = queue_handle.get() {
                drop(queue_handle);

                // prepare task
                let wait_for_finish;
                {
                    let mut task = task_mutex.lock().expect("mutex poisoned");

                    // Start the task and determine if we need to wait for completion
                    wait_for_finish = task.start_task();
                }

                // wait until task is finished if needed
                if wait_for_finish {
                    for _ in 0..10000000 {
                        let mut task = task_mutex.lock().expect("mutex poisoned");
                        if task.is_task_finished() {
                            task.finalize_task();
                            break;
                        }
                        drop(task);
                        thread::sleep(std::time::Duration::from_millis(10));
                    }
                } else {
                    // If no waiting is needed, just finalize the task
                    let mut task = task_mutex.lock().expect("mutex poisoned");
                    task.finalize_task();
                }
            } else {
                drop(queue_handle);
                // No tasks available, sleep for a second before checking again
                thread::sleep(std::time::Duration::from_secs(1));
            }
        }
        log::debug!("Stopped model-thread");
    });

    let mut content = ModelContent {
        model_meta: model_meta,

        hexagon_data: HashMap::new(),
        input_groups: HashMap::new(),
        output_groups: HashMap::new(),

        transfer_block: Arc::new(Mutex::new(TransferBlock::new(&Uuid::new_v4(), model_uuid))),
        start_end_block: Arc::new(Mutex::new(StartEndBlock::new(&Uuid::new_v4(), model_uuid))),

        queue,
        handle: Some(handle),
        running,
    };

    // initialize input-blocks
    let temp_input_copy = content.model_meta.inputs.clone();
    for input_meta in temp_input_copy.iter() {
        content.add_input_group(&input_meta.hexagon_uuid, &input_meta.name)?;
    }

    // initilize output-blocks
    let temp_output_copy = content.model_meta.outputs.clone();
    for output_meta in temp_output_copy.iter() {
        content.add_output_group(
            &output_meta.hexagon_uuid,
            &output_meta.name,
            &OutputType::PlainOutput,
        )?;
    }

    Ok(content)
}

// ==================================================================================================

impl ModelContent {
    pub fn add_core_block(
        &mut self,
        hexagon_uuid: &Uuid,
        block_uuid: &Uuid,
        block_mutex: &Arc<Mutex<CoreBlock>>,
    ) -> Result<(), AinariError> {
        return self.add_block(
            hexagon_uuid,
            block_uuid,
            &(block_mutex.clone() as Arc<Mutex<dyn Block>>),
        );
    }

    pub fn add_output_block(
        &mut self,
        hexagon_uuid: &Uuid,
        block_uuid: &Uuid,
        block_mutex: &Arc<Mutex<OutputBlock>>,
    ) -> Result<(), AinariError> {
        return self.add_block(
            hexagon_uuid,
            block_uuid,
            &(block_mutex.clone() as Arc<Mutex<dyn Block>>),
        );
    }

    pub fn add_input_group(
        &mut self,
        hexagon_uuid: &Uuid,
        group_name: &String,
    ) -> Result<(), AinariError> {
        self.hexagon_data
            .entry(*hexagon_uuid)
            .or_insert_with(|| Arc::new(RwLock::new(HexagonData::new())));

        // check if block with name already exist in the input-list
        if self.input_groups.contains_key(group_name) {
            let msg = format!("Input-group with name '{group_name}' already exist.");
            return Err(AinariError::InvalidInput(msg));
        }

        self.input_groups
            .insert(group_name.clone(), Arc::new(RwLock::new(InputGroup::new())));

        Ok(())
    }

    pub fn add_output_group(
        &mut self,
        hexagon_uuid: &Uuid,
        group_name: &String,
        output_type: &OutputType,
    ) -> Result<(), AinariError> {
        self.hexagon_data
            .entry(*hexagon_uuid)
            .or_insert_with(|| Arc::new(RwLock::new(HexagonData::new())));

        if self.output_groups.contains_key(group_name) {
            let msg = format!("Output-group with name '{group_name}' already exist.");
            return Err(AinariError::InvalidInput(msg));
        }

        self.output_groups.insert(
            group_name.clone(),
            Arc::new(RwLock::new(OutputGroup::new(output_type))),
        );

        Ok(())
    }

    fn add_block(
        &mut self,
        hexagon_uuid: &Uuid,
        block_uuid: &Uuid,
        block_mutex: &Arc<Mutex<dyn Block>>,
    ) -> Result<(), AinariError> {
        // get hexagon from model
        self.hexagon_data
            .entry(*hexagon_uuid)
            .or_insert_with(|| Arc::new(RwLock::new(HexagonData::new())));

        let mut hexgon_link = if let Some(h) = self.hexagon_data.get_mut(hexagon_uuid) {
            h.write().expect("mutex poisoned")
        } else {
            let msg = format!("Hexagon with uuid '{hexagon_uuid}' not found.");
            return Err(AinariError::InvalidInput(msg));
        };

        // add new block
        if hexgon_link.blocks.contains_key(block_uuid) {
            let msg = format!("Block with uuid '{block_uuid}' already exist.");
            return Err(AinariError::InvalidInput(msg));
        }

        hexgon_link
            .blocks
            .insert(*block_uuid, Arc::clone(block_mutex));
        Ok(())
    }

    pub fn get_input_group(&self, name: &String) -> Result<Arc<RwLock<InputGroup>>, AinariError> {
        if let Some(input_block_mutex) = self.input_groups.get(name) {
            Ok(input_block_mutex.clone())
        } else {
            let msg = format!("Input-Group with name '{name}' not found.");
            Err(AinariError::InvalidInput(msg))
        }
    }

    pub fn get_output_group(&self, name: &String) -> Result<Arc<RwLock<OutputGroup>>, AinariError> {
        if let Some(output_block_mutex) = self.output_groups.get(name) {
            Ok(output_block_mutex.clone())
        } else {
            let msg = format!("Output-Block with name '{name}' not found.");
            Err(AinariError::InvalidInput(msg))
        }
    }

    pub fn apply_input(&self, name: &String, values: &[f32]) -> Result<(), AinariError> {
        Ok(())
    }

    fn write_output_into_dataset(
        &self,
        model_uuid: &Uuid,
        file_handle: &mut DataSetFileWriteHandle,
    ) -> Result<(), AinariError> {
        // get column-description from the dataset
        for (name, col_get) in &file_handle.header.columns {
            let size_output = (col_get.end - col_get.start) as usize;
            let mut output_read = vec![0.0f32; size_output];

            let output_group_mutex = self.get_output_group(name)?;

            let mut output_group = output_group_mutex.read().expect("mutex poisoned");
            convert_output_to_buffer(&mut output_read, &output_group);

            let output_bytes = cast_slice(&output_read);
            let _ = file_handle.target_file.write_all(output_bytes);
        }

        Ok(())
    }

    /// Stops the model's worker thread.
    ///
    /// This method sets the running flag to false and joins the worker thread.
    pub fn stop(&mut self) {
        // remove all open tasks from the queue
        let mut queue_handle = self.queue.lock().expect("mutex poisoned");
        queue_handle.clear();
        drop(queue_handle);

        thread::sleep(std::time::Duration::from_millis(5));

        // stop all threads
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Adds a task to the model's task queue.
    ///
    /// # Arguments
    ///
    /// * `task` - The task to be added to the queue
    pub fn add_task(&mut self, task: Task) {
        let mut queue_handle = self.queue.lock().expect("mutex poisoned");
        queue_handle.add(task);
    }

    /// Gets the number of open tasks in the queue.
    ///
    /// # Returns
    ///
    /// The number of tasks currently in the queue
    pub fn get_number_open_tasks(&mut self) -> usize {
        let queue_handle = self.queue.lock().expect("mutex poisoned");
        queue_handle.len()
    }

    /// Processes inputs through the model and returns outputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input names to their corresponding data
    /// * `outputs` - Map of output names to buffers that will be filled with results
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of the operation
    pub fn request(
        &mut self,
        inputs: &HashMap<String, Vec<f32>>,
        outputs: &mut HashMap<String, Vec<f32>>,
    ) -> Result<(), AinariError> {
        // let mut counter = self.finish_counter_mutex.lock().expect("mutex poisoned");
        // let task_compare = counter.output_compare;
        // counter.reset(task_compare, 0);
        // drop(counter);

        // // reset output-values in the backend
        // {
        //     let model_data_handler = MODEL_HANDLER.read().expect("mutex poisoned");
        //     for hexagon_name in outputs.keys() {
        //         let output_buffer_mutex =
        //             model_data_handler.get_output_buffer(&self.model_uuid, hexagon_name)?;
        //         let mut output_buffer = output_buffer_mutex.lock().expect("mutex poisoned");
        //         output_buffer.reset_output();
        //     }
        // }

        // for (hexagon_name, data) in inputs {
        //     tasks::apply_plain_input(
        //         &self.model_uuid,
        //         hexagon_name,
        //         data.as_slice(),
        //         data.len() as u64,
        //         0,
        //         1,
        //         &WorkerTaskType::Process,
        //     )?;
        // }

        // run_iteration(&self.model_uuid)?;

        // // get output-values from the backend
        // let model_data_handler = MODEL_HANDLER.read().expect("mutex poisoned");
        // for (hexagon_name, data) in outputs.iter_mut() {
        //     let output_buffer_mutex =
        //         model_data_handler.get_output_buffer(&self.model_uuid, hexagon_name)?;

        //     let mut output_buffer = output_buffer_mutex.lock().expect("mutex poisoned");
        //     convert_output_to_buffer(data, &mut output_buffer);
        // }

        Ok(())
    }

    /// Trains the model using the provided inputs and expected outputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input names to their corresponding data
    /// * `outputs` - Map of output names to their expected values
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of the operation
    pub fn train(
        &mut self,
        inputs: &HashMap<String, Vec<f32>>,
        outputs: &HashMap<String, Vec<f32>>,
    ) -> Result<(), AinariError> {
        // let mut counter = self.finish_counter_mutex.lock().expect("mutex poisoned");
        // let task_compare = counter.input_compare + counter.output_compare;
        // counter.reset(task_compare, 0);
        // drop(counter);

        for (hexagon_name, data) in outputs {
            // let _ = tasks::apply_expected(
            //     &self.model_uuid,
            //     hexagon_name,
            //     data.as_slice(),
            //     data.len() as u64,
            // );
        }

        for (hexagon_name, data) in inputs {
            // tasks::apply_plain_input(
            //     &self.model_uuid,
            //     hexagon_name,
            //     data.as_slice(),
            //     data.len() as u64,
            //     0,
            //     1,
            //     &WorkerTaskType::Train,
            // )?;
        }

        run_iteration(&self.model_meta.uuid)?;

        Ok(())
    }
}

impl Drop for ModelContent {
    /// Cleanup when the ModelInterface is dropped.
    ///
    /// Ensures the worker thread is stopped before the ModelInterface is destroyed.
    fn drop(&mut self) {
        self.stop(); // make sure to stop thread on drop~!
    }
}

/// Executes a single iteration of model processing.
///
/// This function waits for all tasks to complete or times out after a certain number of iterations.
///
/// # Arguments
///
/// * `model_uuid` - Unique identifier for the model
/// * `finish_counter_mutex` - Shared counter for tracking task completion
///
/// # Returns
///
/// Result indicating success or failure of the operation
fn run_iteration(model_uuid: &Uuid) -> Result<(), AinariError> {
    for _ in 0..10000000 {
        // let finish_counter = finish_counter_mutex.lock().expect("mutex poisoned");
        // if finish_counter.is_finished() {
        //     return Ok(());
        // }
        // drop(finish_counter);
        thread::sleep(std::time::Duration::from_micros(1));
    }

    let msg = format!("Timeout while processing model with uuid {model_uuid}");
    Err(AinariError::InternalError(msg))
}

#[cfg(test)]
mod tests {
    use ainari_model_parser::model_parser::parse_model_template;
    use serial_test::serial;

    use super::*;

    #[test]
    #[serial]
    fn test_create_model() {
        let model_uuid = Uuid::new_v4();
        let name = "test_model".to_string();
        let template = "version: 1 
        settings:
            neuron_cooldown: 1000000000.0;
            refractory_time: 1;
            max_connection_distance: 1;
        hexagons: 
            1,1,1; 
            2,2,2; 
        axons: 
            1,1,1 -> 2,2,2; 
        inputs: 
            key1: 1,1,1; 
        outputs: 
            key2: 2,2,2;"
            .to_string();

        let mut parsed_model = parse_model_template(&name, &template).unwrap();
        parsed_model.uuid = model_uuid;

        let ret = init_new_model(&model_uuid, parsed_model);
        assert!(ret.is_ok());

        let model = ret.unwrap();

        assert_eq!(model.model_meta.uuid, model_uuid);

        assert_eq!(model.hexagon_data.len(), 2);
    }
}
