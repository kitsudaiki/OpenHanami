# libKitsunemimiCpu

## Description

Simple library to read different information of the cpu, like topological information, speed and energy consumption with RAPL, manipulate the speed of single cores of the cpu and read information of the local memory.

## Usage by example

### Get and set cpu-information

The functions of the header https://github.com/kitsudaiki/libKitsunemimiCpu/blob/develop/include/libKitsunemimiCpu/cpu.h should be quite self-explaining.

**Use it at your own risk**

- The threadId is the ID of the CPU-thread. So if you have a CPU with 4 cores and Hyperthreading, then you have the Thread-IDs 0-7 available.
- To change the cpu-frequency, the program has to run as root.
- If you don't have a CPU, which can change the frequency of a core separately, then you have to set the desired new CPU-frequecy for ALL Cpu-Threads to enforce the change of the frequency.

### Power-measureing with RAPL

- **Requirements:**
    - Required specific CPU-architecture:
        - **Intel**:
            - Sandy-Bridge or newer
        - **AMD**:
            - Zen-Architecture or newer
            - for CPUs of AMD Zen/Zen2 Linux-Kernel of version `5.8` or newer must be used, for Zen3 Linux-Kernel of version `5.11` or newer

    - the `msr`-kernel module has to be loaded with `modeprobe msr`.
    - has to be run as root

```cpp
#include <libKitsunemimiCpu/cpu.h>
#include <libKitsunemimiCpu/rapl.h>
#include <libKitsunemimiCpu/memory.h>
#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCommon/threading/thread.h>

Kitsunemimi::Rapl rapl(0);
rapl.initRapl(error);
std::cout<<"info: "<<rapl.getInfo().toString()<<std::endl;

// make firsth request
RaplDiff diff = rapl.calculateDiff();

// sleep 10 seconds
sleep(10);

// make second request
diff = rapl.calculateDiff();
std::cout<<diff.toString()<<std::endl;
// output contains power-consumption per second and total power-consumption within the 10 seconds
```
