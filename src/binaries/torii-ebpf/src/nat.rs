use aya_ebpf::programs::XdpContext;
use network_types::eth::{EthHdr, EtherType};
use network_types::ip::IpProto;
use network_types::icmp::IcmpHdr;
use crate::headers::{Ipv4Hdr, UdpHdr, TcpHdr, ArpHdr};
use crate::utils::{ptr_at_mut, ipv4_checksum, csum_replace4};
use crate::maps::{FIP_DNAT_MAP, FIP_SNAT_MAP};

#[inline(always)]
pub fn apply_dnat(ctx: &XdpContext, eth_type: EtherType) -> Option<u32> {
    if eth_type == EtherType::Ipv4 {
        if let Ok(ipv4_ptr) = ptr_at_mut::<Ipv4Hdr>(ctx, EthHdr::LEN) {
            let mut ipv4 = unsafe { core::ptr::read_unaligned(ipv4_ptr) };
            let mut dst_val = u32::from_be(ipv4.dst_addr);
            
            if let Some(&internal_ip) = unsafe { FIP_DNAT_MAP.get(&dst_val) } {
                ipv4.dst_addr = u32::to_be(internal_ip);
                ipv4.check = 0;
                ipv4.check = ipv4_checksum(&ipv4);
                unsafe { core::ptr::write_unaligned(ipv4_ptr, ipv4) };
                
                let l4_offset = EthHdr::LEN + ((ipv4.version_ihl & 0x0F) * 4) as usize;
                
                if ipv4.protocol == IpProto::Tcp as u8 {
                    if let Ok(tcp) = ptr_at_mut::<TcpHdr>(ctx, l4_offset) {
                        let mut check = unsafe { (*tcp).check };
                        csum_replace4(&mut check, u32::to_be(dst_val), u32::to_be(internal_ip));
                        unsafe { (*tcp).check = check };
                    }
                } else if ipv4.protocol == IpProto::Udp as u8 {
                    if let Ok(udp) = ptr_at_mut::<UdpHdr>(ctx, l4_offset) {
                        let mut check = unsafe { (*udp).check };
                        if check != 0 {
                            csum_replace4(&mut check, u32::to_be(dst_val), u32::to_be(internal_ip));
                            unsafe { (*udp).check = check };
                        }
                    }
                } else if ipv4.protocol == IpProto::Icmp as u8 {
                    if let Ok(_icmp) = ptr_at_mut::<IcmpHdr>(ctx, l4_offset) {
                        // Note: ICMPv4 does NOT include the IP pseudo-header in its checksum.
                        // For 1:1 NAT, changing the outer IP address doesn't invalidate 
                        // the L4 checksum for standard ICMP Echo Requests/Replies.
                        // We check the bounds here to ensure packet validity, but no
                        // csum_replace4 is necessary unless altering the ICMP identifier.
                    }
                }
                dst_val = internal_ip;
            }
            return Some(dst_val);
        }
    } else if eth_type == EtherType::Arp {
        if let Ok(arp_ptr) = ptr_at_mut::<ArpHdr>(ctx, EthHdr::LEN) {
            let mut arp = unsafe { core::ptr::read_unaligned(arp_ptr) };
            let mut tpa_val = u32::from_be(arp.tpa);
            
            if let Some(&internal_ip) = unsafe { FIP_DNAT_MAP.get(&tpa_val) } {
                arp.tpa = u32::to_be(internal_ip);
                unsafe { core::ptr::write_unaligned(arp_ptr, arp) };
                tpa_val = internal_ip;
            }
            return Some(tpa_val);
        }
    }
    None
}

#[inline(always)]
pub fn apply_snat(ctx: &XdpContext, eth_type: EtherType) -> Option<u32> {
    let mut dest_ip = None;
    if eth_type == EtherType::Ipv4 {
        if let Ok(inner_ip_ptr) = ptr_at_mut::<Ipv4Hdr>(ctx, EthHdr::LEN) {
            let mut inner_ip = unsafe { core::ptr::read_unaligned(inner_ip_ptr) };
            dest_ip = Some(u32::from_be(inner_ip.dst_addr));
            
            let src_val = u32::from_be(inner_ip.src_addr);
            
            if let Some(&fip) = unsafe { FIP_SNAT_MAP.get(&src_val) } {
                inner_ip.src_addr = u32::to_be(fip);
                inner_ip.check = 0;
                inner_ip.check = ipv4_checksum(&inner_ip);
                unsafe { core::ptr::write_unaligned(inner_ip_ptr, inner_ip) };
                
                let l4_offset = EthHdr::LEN + ((inner_ip.version_ihl & 0x0F) * 4) as usize;
                
                if inner_ip.protocol == IpProto::Tcp as u8 {
                    if let Ok(tcp) = ptr_at_mut::<TcpHdr>(ctx, l4_offset) {
                        let mut check = unsafe { (*tcp).check };
                        csum_replace4(&mut check, u32::to_be(src_val), u32::to_be(fip));
                        unsafe { (*tcp).check = check };
                    }
                } else if inner_ip.protocol == IpProto::Udp as u8 {
                    if let Ok(udp) = ptr_at_mut::<UdpHdr>(ctx, l4_offset) {
                        let mut check = unsafe { (*udp).check };
                        if check != 0 {
                            csum_replace4(&mut check, u32::to_be(src_val), u32::to_be(fip));
                            unsafe { (*udp).check = check };
                        }
                    }
                } else if inner_ip.protocol == IpProto::Icmp as u8 {
                    if let Ok(_icmp) = ptr_at_mut::<IcmpHdr>(ctx, l4_offset) {
                        // Pass-through without csum rewrite for the same reasons as DNAT.
                    }
                }
            }
        }
    } else if eth_type == EtherType::Arp {
        if let Ok(arp_ptr) = ptr_at_mut::<ArpHdr>(ctx, EthHdr::LEN) {
            let mut arp = unsafe { core::ptr::read_unaligned(arp_ptr) };
            let spa_val = u32::from_be(arp.spa);
            
            if let Some(&fip) = unsafe { FIP_SNAT_MAP.get(&spa_val) } {
                arp.spa = u32::to_be(fip);
                unsafe { core::ptr::write_unaligned(arp_ptr, arp) };
            }
            dest_ip = Some(u32::from_be(arp.tpa));
        }
    }
    dest_ip
}