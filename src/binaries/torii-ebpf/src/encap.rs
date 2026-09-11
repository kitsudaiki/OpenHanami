use aya_ebpf::programs::XdpContext;
use aya_ebpf::bindings::xdp_action;
use network_types::eth::{EthHdr, EtherType};
use network_types::ip::IpProto;
use crate::headers::{Ipv4Hdr, UdpHdr};
use crate::utils::{ptr_at_mut, ipv4_checksum};
use torii_common::RouteTarget;

#[inline(always)]
pub fn encap_and_redirect(ctx: &XdpContext, target: &RouteTarget) -> u32 {
    if unsafe { aya_ebpf::helpers::bpf_xdp_adjust_head(ctx.ctx, -42) } != 0 {
        return xdp_action::XDP_DROP;
    }

    let pkt_len = (ctx.data_end() - ctx.data()) as u16;

    let new_ethhdr = match ptr_at_mut::<EthHdr>(ctx, 0) {
        Ok(h) => h, Err(_) => return xdp_action::XDP_DROP,
    };
    let new_ipv4hdr = match ptr_at_mut::<Ipv4Hdr>(ctx, EthHdr::LEN) {
        Ok(h) => h, Err(_) => return xdp_action::XDP_DROP,
    };
    let new_udphdr = match ptr_at_mut::<UdpHdr>(ctx, EthHdr::LEN + Ipv4Hdr::LEN) {
        Ok(h) => h, Err(_) => return xdp_action::XDP_DROP,
    };

    let mut eth = unsafe { core::ptr::read_unaligned(new_ethhdr) };
    eth.src_addr = target.encap_src_mac;
    eth.dst_addr = target.encap_dst_mac;
    eth.ether_type = EtherType::Ipv4;
    unsafe { core::ptr::write_unaligned(new_ethhdr, eth) };

    let mut ip_hdr = Ipv4Hdr {
        version_ihl: (4 << 4) | 5, tos: 0,
        tot_len: u16::to_be(pkt_len - EthHdr::LEN as u16),
        id: 0, frag_off: 0, ttl: 64, protocol: IpProto::Udp as u8, check: 0,
        src_addr: u32::to_be(target.encap_src_ip),
        dst_addr: u32::to_be(target.encap_dst_ip),
    };
    ip_hdr.check = ipv4_checksum(&ip_hdr);
    unsafe { core::ptr::write_unaligned(new_ipv4hdr, ip_hdr) };

    let udp_hdr = UdpHdr {
        source: u16::to_be(5555), dest: u16::to_be(5555),
        len: u16::to_be(pkt_len - EthHdr::LEN as u16 - Ipv4Hdr::LEN as u16), check: 0,
    };
    unsafe { core::ptr::write_unaligned(new_udphdr, udp_hdr) };

    unsafe { aya_ebpf::helpers::bpf_redirect(target.ifindex, 0) as u32 }
}