use aya_ebpf::macros::map;
use aya_ebpf::maps::HashMap;
use torii_common::RouteTarget;

#[map]
pub static ROUTE_MAP: HashMap<u32, RouteTarget> = HashMap::with_max_entries(1024, 0);

#[map]
pub static FIP_DNAT_MAP: HashMap<u32, u32> = HashMap::with_max_entries(1024, 0);

#[map]
pub static FIP_SNAT_MAP: HashMap<u32, u32> = HashMap::with_max_entries(1024, 0);

#[inline(always)]
pub fn lookup_route(ip: u32) -> Option<RouteTarget> {
    if let Some(target) = unsafe { ROUTE_MAP.get(&ip) } {
        return Some(*target);
    }
    // Fallback to default route (0.0.0.0)
    if let Some(target) = unsafe { ROUTE_MAP.get(&0) } {
        return Some(*target);
    }
    None
}