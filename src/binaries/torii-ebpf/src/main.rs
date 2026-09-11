#![no_std]
#![no_main]

mod headers;
mod utils;
mod maps;
mod nat;
mod encap;
mod decap; // Add the new module

use aya_ebpf::{
    bindings::xdp_action,
    macros::xdp,
    programs::XdpContext,
};
use network_types::eth::EthHdr;

use utils::ptr_at;
use maps::lookup_route;
use nat::apply_dnat;
use encap::encap_and_redirect;
use decap::{is_tunnel_packet, process_tunnel_packet};

#[xdp]
pub fn overlay_ingress(ctx: XdpContext) -> u32 {
    let ethhdr = match ptr_at::<EthHdr>(&ctx, 0) {
        Ok(hdr) => hdr,
        Err(_) => return xdp_action::XDP_PASS,
    };
    
    let eth_type = unsafe { core::ptr::read_unaligned(ethhdr).ether_type };

    // Apply DNAT if necessary. Returns the true Target IP (or original IP if no DNAT).
    if let Some(dest_ip) = apply_dnat(&ctx, eth_type) {
        if let Some(target) = lookup_route(dest_ip) {
            if target.action == 1 { 
                return encap_and_redirect(&ctx, &target);
            } else {
                return unsafe { aya_ebpf::helpers::bpf_redirect(target.ifindex, 0) } as u32;
            }
        }
    }

    xdp_action::XDP_PASS
}

#[xdp]
pub fn underlay_ingress(ctx: XdpContext) -> u32 {
    // If this packet is encapsulated VM traffic, process and route it locally
    if is_tunnel_packet(&ctx) {
        return process_tunnel_packet(&ctx);
    }
    
    // Otherwise, let standard host networking handle it
    xdp_action::XDP_PASS
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! { 
    unsafe { core::hint::unreachable_unchecked() }
}