use cloud_hypervisor_client::apis::DefaultApi;
use cloud_hypervisor_client::models::{
    ConsoleMode, CpusConfig, DiskConfig, MemoryConfig, NetConfig, PayloadConfig, SerialConfig,
    VmConfig,
};
use cloud_hypervisor_client::socket_based_api_client;
use std::process::Command;
use std::time::Duration;

async fn create_instance() -> Result<(), Box<dyn std::error::Error>> {
    let socket_path = "/tmp/cloud-hypervisor.sock";

    // Clean up leftover socket from any previous crashed runs
    let _ = std::fs::remove_file(socket_path);

    // Spawn the VMM
    let mut vmm_process =
        Command::new("/home/neptune/Schreibtisch/Projects/vm_test/cloud-hypervisor")
            .arg("--api-socket")
            .arg(socket_path)
            .spawn()
            .expect("Failed to start cloud-hypervisor binary");

    tokio::time::sleep(Duration::from_millis(500)).await;

    if let Some(status) = vmm_process.try_wait()? {
        eprintln!(
            "CRITICAL ERROR: cloud-hypervisor crashed during startup. Exit status: {}",
            status
        );
        std::process::exit(1);
    }

    let client = socket_based_api_client(socket_path);

    let payload = PayloadConfig {
        // Point this to the full EDK2 UEFI firmware
        firmware: Some(String::from(
            "/home/neptune/Schreibtisch/Projects/vm_test/CLOUDHV.fd",
        )),

        kernel: None,
        cmdline: None,
        initramfs: None,
        ..Default::default()
    };

    let vm_config = VmConfig {
        payload,
        cpus: Some(CpusConfig {
            boot_vcpus: 2,
            max_vcpus: 2,
            ..Default::default()
        }),
        // // Fixed: Using SerialConfig and the strictly-typed ConsoleMode enum
        // serial: Some(SerialConfig {
        //     mode: ConsoleMode::Tty,
        //     ..Default::default()
        // }),
        net: Some(vec![NetConfig {
            tap: Some(String::from("vmtap0")),
            ..Default::default()
        }]),
        memory: Some(MemoryConfig {
            size: 1_073_741_824, // 1 GB RAM
            ..Default::default()
        }),
        disks: Some(vec![
            DiskConfig {
                path: Some(String::from(
                    "/home/neptune/Schreibtisch/Projects/vm_test/ubuntu-24.04.raw",
                )),
                readonly: Some(false),
                ..Default::default()
            },
            DiskConfig {
                path: Some(String::from(
                    "/home/neptune/Schreibtisch/Projects/vm_test/seed.iso",
                )),
                readonly: Some(true),
                ..Default::default()
            },
        ]),
        ..Default::default()
    };

    println!("Creating VM...");
    client
        .create_vm(vm_config)
        .await
        .map_err(|e| format!("Failed to create VM: {:?}", e))?;

    println!("Booting VM...");
    client
        .boot_vm()
        .await
        .map_err(|e| format!("Failed to boot VM: {:?}", e))?;

    println!("VM is running! You should see the boot logs below.");

    // Wait for the VM process to exit natively
    vmm_process.wait()?;
    Ok(())
}
