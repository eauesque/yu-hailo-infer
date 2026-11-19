//! Test-only controls, invoked on the same device thread as the stub.
extern "C" {
    fn yu_hailort_stub_wd_fail(stage: i32);
    fn yu_hailort_stub_wd_metadata(field: i32, value: f64);
    fn yu_hailort_stub_wd_output(bytes: *const u8, size: usize, delay_ms: u32);
    fn yu_hailort_stub_wd_count(event: usize) -> usize;
    fn yu_hailort_stub_wd_timeout() -> u32;
    fn yu_hailort_stub_wd_input(bytes: *mut u8, size: usize) -> usize;
}

pub(crate) fn setup(fields: &[(i32, f64)], failure: i32, delay_ms: u32) {
    // SAFETY: thread-local stub controls; the output bytes are copied synchronously.
    unsafe {
        yu_hailort_stub_wd_metadata(-1, 0.0);
        for &(field, value) in fields {
            yu_hailort_stub_wd_metadata(field, value);
        }
        yu_hailort_stub_wd_fail(failure);
        yu_hailort_stub_wd_output([5, 19, 18].as_ptr(), 3, delay_ms);
    }
}

pub(crate) fn configure_count() -> usize {
    // SAFETY: event 3 is the stub's configure counter.
    unsafe { yu_hailort_stub_wd_count(3) }
}

pub(crate) fn run_count() -> usize {
    // SAFETY: event 5 counts entry to the stub's WD run on this device thread.
    unsafe { yu_hailort_stub_wd_count(5) }
}

pub(crate) fn last_run() -> (u32, Vec<u8>) {
    let mut input = vec![0; 602_112];
    // SAFETY: the stub copies only when the supplied buffer has exactly the recorded size.
    unsafe {
        let size = yu_hailort_stub_wd_input(input.as_mut_ptr(), input.len());
        input.truncate(size);
        (yu_hailort_stub_wd_timeout(), input)
    }
}
