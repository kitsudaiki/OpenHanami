use aya::maps::{HashMap as AyaHashMap, MapData};
use aya::programs::{Xdp, XdpFlags};
use aya::{include_bytes_aligned, Bpf};
use torii_common::RouteTarget;
use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::Mutex;
use uuid::Uuid;

use ainari_api_structs::route_structs::*;

// --- Application State ---
pub struct GatewayState {
    pub routes: HashMap<String, Route>,
    pub floating_ips: HashMap<String, String>,
    pub route_map: AyaHashMap<MapData, u32, RouteTargetPod>,
    pub fip_dnat_map: AyaHashMap<MapData, u32, u32>,
    pub fip_snat_map: AyaHashMap<MapData, u32, u32>,
}


lazy_static::lazy_static! {
    pub static ref ROUTE_HANDLER: Arc<Mutex<GatewayState>> = Arc::new(Mutex::new(init_routing()));
}

pub fn init_routing() -> GatewayState {
    let overlay_iface = std::env::var("OVERLAY_IFACE").unwrap_or_else(|_| "veth-gw".to_string());
    let underlay_iface = std::env::var("UNDERLAY_IFACE").unwrap_or_else(|_| "eth0".to_string());

    let mut bpf = Bpf::load(include_bytes_aligned!(concat!(env!("OUT_DIR"), "/torii"))).unwrap();

    let route_map_data = bpf.take_map("ROUTE_MAP").expect("Missing ROUTE_MAP");
    let route_map: AyaHashMap<_, u32, RouteTargetPod> = AyaHashMap::try_from(route_map_data).unwrap();

    let fip_dnat_map_data = bpf.take_map("FIP_DNAT_MAP").expect("Missing FIP_DNAT_MAP");
    let fip_dnat_map: AyaHashMap<_, u32, u32> = AyaHashMap::try_from(fip_dnat_map_data).unwrap();

    let fip_snat_map_data = bpf.take_map("FIP_SNAT_MAP").expect("Missing FIP_SNAT_MAP");
    let fip_snat_map: AyaHashMap<_, u32, u32> = AyaHashMap::try_from(fip_snat_map_data).unwrap();

    let overlay: &mut Xdp = bpf.program_mut("overlay_ingress").unwrap().try_into().unwrap();
    overlay.load().unwrap();
    overlay.attach(&overlay_iface, XdpFlags::SKB_MODE).unwrap();

    let underlay: &mut Xdp = bpf.program_mut("underlay_ingress").unwrap().try_into().unwrap();
    underlay.load().unwrap();
    underlay.attach(&underlay_iface, XdpFlags::SKB_MODE).unwrap();

    let state = GatewayState {
        routes: HashMap::new(),
        floating_ips: HashMap::new(),
        route_map,
        fip_dnat_map,
        fip_snat_map,
    };

    state
}


#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct RouteTargetPod(pub RouteTarget);

#[allow(unsafe_attr_outside_unsafe)]
unsafe impl aya::Pod for RouteTargetPod {}

