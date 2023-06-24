# Monitoring

Their are two major tasks to do:

1. read and observe system-information of CPU 

    !!! info

        Observation of memory, GPU, network and storage comes later.

2. optimize the usage of the hardware by supporting CPU-scheduling of all threads of all components

3. Change CPU-frequency based on the workload of Kyouko

4. measure Power-consumption, thermal output and frequency of the CPU

    !!! info "Forecast"

        These measurements will be used in the future in order to optimize the binding of CPU-threads and modification of the CPU-frequency.


## **Collecting information**

For collecting of hardware-information the library `libKitsunemimiSakuraHardware` is used and interact with other specific libraries in order to get all information of the hardware.

![Workflow-component](../img/Azuki_hardware_layer.drawio)

At the moment only CPU related information are collected. The collected data are hold in the data-structure for easier access. In this structure it is easier to identify for example, which two CPU-threads are sharing the same physical CPU-core.

## **Thread-Binding**

The only energy-optimization, at least for now, is thread-binding.

Each thread, which is created with the thread-class of the library `libKitsunemimiCommon`, has a name attached. With this name the thread is accessible over the singleton thread-manager-class, which each component contains. Over a internal endpoint, which is provided by `libAzukiHeart` this can be used to access over the network to all of theses threads of each component. 
Additionally the thread-class possess the ability to bind the thread to one or more CPU-threads to enforce the location on the CPU, where the thread should be executed. Together with the network-access to the threads over the thread-manager-class, Azuki can define remote for each thread, on which CPU-thread and -core it should executed.

![Workflow-component](../img/Azuki_internal.drawio)

In the current version this is quite static and simple. Each processing-thread of Kyouko, which process `Cluster` and `Segments`, runs on the CPU-cores >1 and every other thread of every component is executed on the two thread of the first physical CPU-core.

By choosing the CPU-threads by Azuki it should be ensured, that not more physical CPU-core are in use, then necessary. By enforcing that the 2 threads of the same physical CPU-core are used, instead of 2 threads on two different cores, it can save much energy. 


## **Controlling CPU frequency**

Azuki can change the frequency of the CPU. This is triggered by Kyouko. Whenever Kyouko has no tasks or open websocket-connections, it send a message to Azuki to reduce the frequency of the CPU in order to reduce power consumption in Idle-state.

!!! warning

    This is only a PoC. Whenever Kyouko has runs in an empty Task-Queue, it orders Azuki to reduce the CPU-frequency to the absolute minimum, which is supported by the CPU. At the moment this can not handle multiple parallel clusters. There is no check if another queue has still some task and so they would interfere each other. Because of this, the feature is not enabled in the version, which is installed by the kubernetes-installation.

The result looks like this:

![Workflow-component](../img/cpu_power.png)

Every spike in the graph means, that a task comes to be processed. Between the task, the cpu is enforced to minimal cpu-frequency and so also minimal power consumption.