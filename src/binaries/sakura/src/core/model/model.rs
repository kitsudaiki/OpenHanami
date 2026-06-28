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

use core::result::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use uuid::Uuid;

use ainari_common::error::AinariError;
use ainari_model_parser::model_meta_structs::*;

use crate::core::model::blocks::end_block::*;
use crate::core::model::blocks::hexagon_block::HexagonBlock;
use crate::core::model::blocks::start_block::*;
use crate::core::processing::task_queue::{TaskHandler, init_task_queue};
use crate::core::processing::tasks::Task;

// ==================================================================================================

pub struct Model {
    pub uuid: Uuid,
    #[allow(dead_code)]
    pub name: String,
    #[allow(dead_code)]
    pub settings: Settings,

    pub start_blocks: HashMap<String, Arc<Mutex<StartBlock>>>,
    pub end_block: Arc<Mutex<EndBlock>>,

    pub task_handler: Arc<Mutex<TaskHandler>>,

    pub thread_handle: Option<JoinHandle<()>>,
    pub running: Arc<AtomicBool>,
}

// ==================================================================================================

impl Model {
    pub fn new(model_meta: &ModelMeta) -> Result<Self, AinariError> {
        let end_hexagon = Arc::new(Mutex::new(HexagonBlock::new(&model_meta.uuid, None)));
        let task_handler = Arc::new(Mutex::new(init_task_queue()));
        let mut new_model = Model {
            uuid: model_meta.uuid,
            name: model_meta.name.clone(),
            settings: model_meta.settings.clone(),

            start_blocks: HashMap::new(),

            end_block: Arc::new(Mutex::new(EndBlock::new(
                &model_meta.uuid,
                end_hexagon,
                &task_handler,
                0,
            ))),

            task_handler,
            thread_handle: None,
            running: Arc::new(AtomicBool::new(true)),
        };

        for input_meta in model_meta.inputs.iter() {
            new_model.add_start_block(&input_meta.name)?;
        }

        for hexagon in model_meta.hexagons.iter() {
            let mut hexagon_block = HexagonBlock::new(&model_meta.uuid, None);
        }

        Ok(new_model)
    }
}

/// Initializes a new model with the given metadata and UUID.
///
/// This creates a complete model structure including all blocks, inputs, and outputs.
///
/// # Arguments
/// * `model_meta` - Metadata describing the model's structure and configuration.
///
/// # Returns
/// * `Ok(())` on success.
/// * `Err(AinariError)` if the model already exists or if initialization fails.
pub fn init_new_model(
    model_meta: ModelMeta,
) -> Result<Arc<Mutex<Model>>, AinariError> {

    let new_model = Model::new(&model_meta)?;

    // clone objects for the 
    let running_clone = Arc::clone(&new_model.running);
    let task_handler_clone = Arc::clone(&new_model.task_handler);
    let model_uuid_clone = model_meta.uuid.clone();
    let new_model_mutex = Arc::new(Mutex::new(new_model));
    let new_model_mutex_clone = new_model_mutex.clone();

    let thread_handle = thread::spawn(move || {
        log::debug!("Started model-thread");
        while running_clone.load(Ordering::Relaxed) {
            // get task from the task-queue and process the task, otherwise sleep until the next check
            let mut task_handler = task_handler_clone.lock().expect("mutex poisoned");
            if let (id, Some(task_mutex)) = task_handler.get_next_from_queue() {
                drop(task_handler);

                // prepare task
                let wait_for_finish;
                {
                    let mut task = task_mutex.lock().expect("mutex poisoned");

                    // Start the task and determine if we need to wait for completion
                    wait_for_finish =
                        match task.start_task(&model_uuid_clone, &new_model_mutex_clone) {
                            Ok(wait_for_finish) => wait_for_finish,
                            Err(_) => {
                                // TODO: error-handling
                                false
                            }
                        };
                }

                // wait until task is finished if needed
                if wait_for_finish {
                    for _ in 0..10000000 {
                        let mut task = task_mutex.lock().expect("mutex poisoned");
                        if task.is_task_finished() {
                            task.finalize_task();
                            task_handler = task_handler_clone.lock().expect("mutex poisoned");
                            let _ = task_handler.finish_process(&id);
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
                drop(task_handler);
                // No tasks available, sleep for a second before checking again
                thread::sleep(std::time::Duration::from_secs(1));
            }
        }
        log::debug!("Stopped model-thread");
    });

    let mut model = new_model_mutex.lock().expect("mutex poisoned");
    model.thread_handle = Some(thread_handle);
    drop(model);

    Ok(new_model_mutex)
}

// ==================================================================================================

impl Model {
    fn add_start_block(
        &mut self,
        group_name: &String,
    ) -> Result<Arc<Mutex<StartBlock>>, AinariError> {
        // check if block with name already exist in the input-list
        if self.start_blocks.contains_key(group_name) {
            let msg = format!("Input-group with name '{group_name}' already exist.");
            return Err(AinariError::InvalidInput(msg));
        }

        let start_hexagon = Arc::new(Mutex::new(HexagonBlock::new(&self.uuid, None)));

        let block_mutex = Arc::new(Mutex::new(StartBlock::new(
            &self.uuid,
            start_hexagon,
            &self.task_handler.clone(),
        )));
        self.start_blocks
            .insert(group_name.clone(), block_mutex.clone());

        Ok(block_mutex)
    }

    pub fn get_start_block(&self, name: &String) -> Result<Arc<Mutex<StartBlock>>, AinariError> {
        if let Some(start_block_mutex) = self.start_blocks.get(name) {
            Ok(start_block_mutex.clone())
        } else {
            let msg = format!("Start-block with name '{name}' not found.");
            Err(AinariError::InvalidInput(msg))
        }
    }

    pub fn get_end_block(&self) -> Arc<Mutex<EndBlock>> {
        self.end_block.clone()
    }

    /// Stops the model's worker thread.
    ///
    /// This method sets the running flag to false and joins the worker thread.
    pub fn stop(&mut self) {
        // remove all open tasks from the queue
        let mut queue_handle = self.task_handler.lock().expect("mutex poisoned");
        queue_handle.clear();
        drop(queue_handle);

        thread::sleep(std::time::Duration::from_millis(5));

        // stop all threads
        self.running.store(false, Ordering::Relaxed);
        if let Some(thread_handle) = self.thread_handle.take() {
            let _ = thread_handle.join();
        }
    }

    /// Adds a task to the model's task queue.
    ///
    /// # Arguments
    ///
    /// * `task` - The task to be added to the queue
    pub fn add_task(&mut self, task: Task) {
        let mut queue_handle = self.task_handler.lock().expect("mutex poisoned");
        queue_handle.add_to_queue(task);
    }

    /// Gets the number of open tasks in the queue.
    ///
    /// # Returns
    ///
    /// The number of tasks currently in the queue
    pub fn get_number_open_tasks(&mut self) -> usize {
        let queue_handle = self.task_handler.lock().expect("mutex poisoned");
        queue_handle.queue_len()
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
        _: &HashMap<String, Vec<f32>>,
        _: &mut HashMap<String, Vec<f32>>,
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
        _: &HashMap<String, Vec<f32>>,
        _: &HashMap<String, Vec<f32>>,
    ) -> Result<(), AinariError> {
        // let mut counter = self.finish_counter_mutex.lock().expect("mutex poisoned");
        // let task_compare = counter.input_compare + counter.output_compare;
        // counter.reset(task_compare, 0);
        // drop(counter);

        // for (hexagon_name, data) in outputs {
        //     // let _ = tasks::apply_expected(
        //     //     &self.model_uuid,
        //     //     hexagon_name,
        //     //     data.as_slice(),
        //     //     data.len() as u64,
        //     // );
        // }

        // for (hexagon_name, data) in inputs {
        //     // tasks::apply_plain_input(
        //     //     &self.model_uuid,
        //     //     hexagon_name,
        //     //     data.as_slice(),
        //     //     data.len() as u64,
        //     //     0,
        //     //     1,
        //     //     &WorkerTaskType::Train,
        //     // )?;
        // }

        Ok(())
    }
}

impl Drop for Model {
    /// Cleanup when the ModelInterface is dropped.
    ///
    /// Ensures the worker thread is stopped before the ModelInterface is destroyed.
    fn drop(&mut self) {
        self.stop(); // make sure to stop thread on drop~!
    }
}

// #[cfg(test)]
// mod tests {
//     use ainari_model_parser::model_parser::parse_model_template;
//     use serial_test::serial;

//     use super::*;

//     #[test]
//     #[serial]
//     fn test_create_model() {
//         let model_uuid = Uuid::new_v4();
//         let name = "test_model".to_string();
//         let template = "version: 1
//         settings:
//             neuron_cooldown: 1000000000.0;
//             refractory_time: 1;
//             max_connection_distance: 1;
//         hexagons:
//             1,1,1;
//             2,2,2;
//         axons:
//             1,1,1 -> 2,2,2;
//         inputs:
//             key1: 1,1,1;
//         outputs:
//             key2: 2,2,2;"
//             .to_string();

//         let mut parsed_model = parse_model_template(&name, &template).unwrap();
//         parsed_model.uuid = model_uuid;

//         let ret = init_new_model(&model_uuid, parsed_model);
//         assert!(ret.is_ok());

//         let model = ret.unwrap();

//         assert_eq!(model.model_meta.uuid, model_uuid);

//         assert_eq!(model.hexagon_data.len(), 2);
//     }
// }
