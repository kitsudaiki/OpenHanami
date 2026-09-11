use aya_ebpf::programs::XdpContext;
use core::mem;
use crate::headers::Ipv4Hdr;

#[inline(always)]
pub fn ptr_at<T>(ctx: &XdpContext, offset: usize) -> Result<*const T, ()> {
    let start = ctx.data();
    let end = ctx.data_end();
    let len = mem::size_of::<T>();
    if start + offset + len > end { return Err(()); }
    Ok((start + offset) as *const T)
}

#[inline(always)]
pub fn ptr_at_mut<T>(ctx: &XdpContext, offset: usize) -> Result<*mut T, ()> {
    let start = ctx.data();
    let end = ctx.data_end();
    let len = core::mem::size_of::<T>();
    if start + offset + len > end { return Err(()); }
    Ok((start + offset) as *mut T)
}

#[inline(always)]
pub fn ipv4_checksum(hdr: &Ipv4Hdr) -> u16 {
    let mut sum: u32 = 0;
    let b = unsafe { core::slice::from_raw_parts(hdr as *const _ as *const u8, 20) };
    
    sum += u16::from_be_bytes([b[0], b[1]]) as u32;
    sum += u16::from_be_bytes([b[2], b[3]]) as u32;
    sum += u16::from_be_bytes([b[4], b[5]]) as u32;
    sum += u16::from_be_bytes([b[6], b[7]]) as u32;
    sum += u16::from_be_bytes([b[8], b[9]]) as u32;
    sum += u16::from_be_bytes([b[10], b[11]]) as u32;
    sum += u16::from_be_bytes([b[12], b[13]]) as u32;
    sum += u16::from_be_bytes([b[14], b[15]]) as u32;
    sum += u16::from_be_bytes([b[16], b[17]]) as u32;
    sum += u16::from_be_bytes([b[18], b[19]]) as u32;
    
    sum = (sum & 0xffff) + (sum >> 16);
    sum = (sum & 0xffff) + (sum >> 16);
    
    u16::to_be(!(sum as u16))
}

#[inline(always)]
pub fn csum_replace4(csum: &mut u16, old: u32, new: u32) {
    let mut sum: u32 = (!*csum & 0xffff) as u32;
    
    let old_1 = (old >> 16) as u16;
    let old_2 = (old & 0xffff) as u16;
    let new_1 = (new >> 16) as u16;
    let new_2 = (new & 0xffff) as u16;
    
    sum = sum.wrapping_add(!old_1 as u32).wrapping_add(!old_2 as u32)
             .wrapping_add(new_1 as u32).wrapping_add(new_2 as u32);
    
    sum = (sum & 0xffff) + (sum >> 16);
    sum = (sum & 0xffff) + (sum >> 16);
    *csum = !(sum as u16);
}