use aya_ebpf::programs::XdpContext;
use aya_ebpf::bindings::xdp_action;
use network_types::eth::{EthHdr, EtherType};
use network_types::ip::IpProto;
use crate::headers::{Ipv4Hdr, UdpHdr};
use crate::utils::ptr_at;
use crate::nat::apply_snat;
use crate::maps::lookup_route;

/// Checks if the incoming packet is a UDP tunnel packet on port 5555
#[inline(always)]
pub fn is_tunnel_packet(ctx: &XdpContext) -> bool {
    let ethhdr = match ptr_at::<EthHdr>(ctx, 0) {
        Ok(hdr) => hdr,
        Err(_) => return false,
    };

    if unsafe { core::ptr::read_unaligned(ethhdr).ether_type } != EtherType::Ipv4 {
        return false;
    }

    let ipv4hdr = match ptr_at::<Ipv4Hdr>(ctx, EthHdr::LEN) {
        Ok(hdr) => hdr,
        Err(_) => return false,
    };

    if unsafe { core::ptr::read_unaligned(ipv4hdr).protocol } != IpProto::Udp as u8 {
        return false;
    }

    let udphdr = match ptr_at::<UdpHdr>(ctx, EthHdr::LEN + Ipv4Hdr::LEN) {
        Ok(hdr) => hdr,
        Err(_) => return false,
    };

    // FIX: Read into a variable first to avoid parser ambiguity
    let dest_port = unsafe { core::ptr::read_unaligned(udphdr).dest };
    dest_port == u16::to_be(5555)
}

/// Strips the outer headers, applies SNAT, and redirects the inner payload
#[inline(always)]
pub fn process_tunnel_packet(ctx: &XdpContext) -> u32 {
    // Strip Outer Eth, IPv4, and UDP headers (14 + 20 + 8 = 42 bytes)
    if unsafe { aya_ebpf::helpers::bpf_xdp_adjust_head(ctx.ctx, 42) } != 0 {
        return xdp_action::XDP_DROP;
    }

    // Parse the decapsulated Inner Ethernet frame
    let inner_eth = match ptr_at::<EthHdr>(ctx, 0) {
        Ok(hdr) => hdr,
        Err(_) => return xdp_action::XDP_DROP,
    };
    
    let eth_type = unsafe { core::ptr::read_unaligned(inner_eth).ether_type };
    
    // Apply SNAT if necessary. Returns the true Target IP
    if let Some(dest_ip) = apply_snat(ctx, eth_type) {
        if let Some(target) = lookup_route(dest_ip) {
            if target.action == 0 { 
                return unsafe { aya_ebpf::helpers::bpf_redirect(target.ifindex, 0) } as u32;
            }
        }
    }
    
    // Drop invalid tunnel packets
    xdp_action::XDP_DROP
}