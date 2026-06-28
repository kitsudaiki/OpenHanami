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

use core_affinity;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use ainari_common::error::AinariError;

use super::worker_queue::*;

/// Represents a worker thread that processes tasks from the worker queue.
///
/// The worker thread runs on a specific CPU core and processes tasks until stopped.
/// It handles different types of tasks including training, processing, and backpropagation.
pub struct WorkerThread {
    /// Unique identifier for the thread
    #[allow(dead_code)]
    pub thread_id: usize,

    /// Handle to the spawned thread
    pub handle: Option<JoinHandle<()>>,

    /// Atomic boolean flag to control the thread's running state
    pub running: Arc<AtomicBool>,
}

fn process_task(worker_task: &WorkerTask) -> Result<bool, AinariError> {
    let mut block = worker_task.block.lock().expect("mutex poisoned");
    let completely_done = block.process()?;

    Ok(completely_done)
}

impl WorkerThread {
    /// Creates a new WorkerThread instance.
    ///
    /// This function initializes a new worker thread on the specified CPU core.
    /// The thread will process tasks from the worker queue until it is stopped.
    ///
    /// # Arguments
    /// * `thread_id` - The ID of the CPU core to run the thread on
    ///
    /// # Returns
    /// * `WorkerThread` - A new WorkerThread instance
    pub fn new(thread_id: usize) -> Self {
        // Create an atomic boolean to control the thread's running state
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);
        log::debug!("Create worker-thread on cpu-thread {thread_id}");

        // Spawn the worker thread
        let handle = thread::spawn(move || {
            log::debug!("Started worker-thread on cpu-thread {thread_id}");

            // Set CPU affinity to pin the thread to the specified core
            let core_id = core_affinity::CoreId { id: thread_id };
            let res = core_affinity::set_for_current(core_id);
            if !res {
                log::warn!("Failed to pin worker-thread to cpu-thread {thread_id}");
            }

            // Main thread loop
            while running_clone.load(Ordering::Relaxed) {
                // Get a task from the worker queue
                let mut worker_queue = WORKER_QUEUE.lock().expect("mutex poisoned");
                if let Some(worker_task) = worker_queue.get() {
                    drop(worker_queue);

                    // Process the task and handle any errors
                    let completely_done = match process_task(&worker_task) {
                        Ok(completely_done) => completely_done,
                        Err(AinariError::Unauthorized(msg)) => {
                            log::error!("{msg}");
                            // TODO: better error-handling
                            true
                        }
                        Err(AinariError::InvalidInput(msg)) => {
                            log::error!("{msg}");
                            // TODO: better error-handling
                            true
                        }
                        Err(AinariError::InternalError(msg)) => {
                            log::error!("{msg}");
                            // TODO: better error-handling
                            true
                        }
                    };

                    // re-queue task, if not completely done
                    if !completely_done {
                        worker_queue = WORKER_QUEUE.lock().expect("mutex poisoned");
                        worker_queue.add(worker_task);
                    }
                } else {
                    drop(worker_queue);
                    // Sleep briefly if there are no tasks to process
                    thread::sleep(Duration::from_millis(1));
                }
            }

            log::debug!("Stopped worker-thread on cpu-thread {thread_id}");
        });

        // Return the new WorkerThread instance
        WorkerThread {
            thread_id,
            handle: Some(handle),
            running,
        }
    }

    /// Stops the worker thread.
    ///
    /// This function sets the running flag to false and waits for the thread to finish execution.
    pub fn stop(&mut self) {
        // Set the running flag to false
        self.running.store(false, Ordering::Relaxed);

        // Wait for the thread to finish
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for WorkerThread {
    /// Ensures the worker thread is stopped when the WorkerThread instance is dropped.
    fn drop(&mut self) {
        self.stop(); // make sure to stop thread on drop~!
    }
}
