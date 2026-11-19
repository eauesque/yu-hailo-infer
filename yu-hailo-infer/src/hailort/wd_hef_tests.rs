use super::*;
use std::io::{Seek, Write};
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

thread_local! {
    static READ_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static OPEN_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static BEFORE_OPEN: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const {
        std::cell::RefCell::new(None)
    };
    static BEFORE_READ: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const {
        std::cell::RefCell::new(None)
    };
}

pub(super) fn before_open() {
    OPEN_ATTEMPTS.with(|count| count.set(count.get() + 1));
    BEFORE_OPEN.with(|hook| {
        if let Some(check) = hook.borrow_mut().take() {
            check();
        }
    });
}

pub(super) fn before_read() {
    READ_ATTEMPTS.with(|count| count.set(count.get() + 1));
    BEFORE_READ.with(|hook| {
        if let Some(check) = hook.borrow_mut().take() {
            check();
        }
    });
}

pub(crate) struct TempDir(pub(crate) PathBuf);

impl TempDir {
    pub(crate) fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "wd-hef-t8-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    pub(crate) fn file(&self) -> PathBuf {
        let path = self.0.join("model.hef");
        std::fs::write(&path, [7, 8, 9]).unwrap();
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn assert_invalid(path: &Path) {
    assert!(matches!(CheckedHef::open(path), Err(WdHefError::Invalid)));
}

#[test]
fn open_flags_include_nofollow() {
    assert_eq!(OPEN_FLAGS & libc::O_NOFOLLOW, libc::O_NOFOLLOW);
}

#[test]
fn revfix_empty_hef_rejected_before_read() {
    let dir = TempDir::new();
    let path = dir.file();
    std::fs::write(&path, []).unwrap();
    assert_invalid(&path);
}

#[test]
fn revfix_device_rejected_before_open() {
    let before = OPEN_ATTEMPTS.with(|count| count.get());
    assert_invalid(Path::new("/dev/null"));
    assert_eq!(OPEN_ATTEMPTS.with(|count| count.get()) - before, 0);
}

#[test]
fn revfix_replacement_between_lstat_and_open_rejected() {
    let dir = TempDir::new();
    let path = dir.file();
    let replacement = path.clone();
    BEFORE_OPEN.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move || {
            std::fs::rename(&replacement, replacement.with_extension("old")).unwrap();
            std::fs::write(&replacement, [1, 2, 3]).unwrap();
        }))
    });
    assert_invalid(&path);
}

#[test]
fn revfix_open_flags_include_noctty() {
    assert_eq!(OPEN_FLAGS & libc::O_NOCTTY, libc::O_NOCTTY);
}

#[cfg(hailo_stub)]
#[test]
fn revfix_timeout_releases_before_buffers_drop_and_evicts() {
    if isolated("hailort::wd_hef::tests::revfix_timeout_releases_before_buffers_drop_and_evicts") {
        return;
    }
    struct Buffer {
        bytes: Vec<u8>,
        drops: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
    }
    impl Drop for Buffer {
        fn drop(&mut self) {
            self.drops.borrow_mut().push(calls());
        }
    }
    let dir = TempDir::new();
    let path = dir.file();
    let cache = WdHefCache::default();
    let drops = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    fail(7);
    let result = cache.with_model(
        &path,
        digest(&path),
        |_, _| Ok(()),
        |model| {
            let input = Buffer {
                bytes: vec![7; 602_112],
                drops: drops.clone(),
            };
            let mut output = Buffer {
                bytes: vec![0; 10_861],
                drops: drops.clone(),
            };
            model.run(&input.bytes, &mut [("output", &mut output.bytes)], 1000)?;
            Ok(())
        },
    );
    assert!(matches!(result, Err(WdHefError::Runtime(WdError::Timeout))));
    assert_eq!(
        *drops.borrow(),
        vec!["open,metadata,configure,run,release"; 2]
    );
    assert!(cache.slot.lock().unwrap().is_none());
    fail(0);
    load(&cache, &path).unwrap();
    assert_eq!(counts()[1], 2);
}

#[test]
fn fifo_rejected_without_blocking() {
    let dir = TempDir::new();
    let path = dir.0.join("fifo");
    assert!(std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .unwrap()
        .success());
    let (send, recv) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        send.send(matches!(CheckedHef::open(&path), Err(WdHefError::Invalid)))
            .unwrap();
    });
    assert!(recv
        .recv_timeout(Duration::from_secs(2))
        .expect("FIFO open blocked"));
}

#[test]
fn directory_rejected() {
    let dir = TempDir::new();
    assert_invalid(&dir.0);
}

#[test]
fn device_rejected() {
    assert_invalid(Path::new("/dev/null"));
}

#[test]
fn terminal_symlink_rejected() {
    let dir = TempDir::new();
    let path = dir.file();
    let link = dir.0.join("link");
    symlink(path, &link).unwrap();
    assert_invalid(&link);
}

#[test]
fn socket_rejected() {
    let dir = TempDir::new();
    let path = dir.0.join("socket");
    let _socket = std::os::unix::net::UnixListener::bind(&path).unwrap();
    assert_invalid(&path);
}

#[test]
fn oversized_sparse_file_rejected_before_read() {
    let dir = TempDir::new();
    let path = dir.file();
    File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(LIMIT + 1)
        .unwrap();
    assert_invalid(&path);
}

#[test]
fn valid_bytes_use_checked_fd_even_after_path_replacement() {
    let dir = TempDir::new();
    let path = dir.file();
    let checked = CheckedHef::open(&path).unwrap();
    std::fs::rename(&path, dir.0.join("old")).unwrap();
    std::fs::write(&path, [10]).unwrap();
    assert_eq!(
        read_bounded(checked.file, LIMIT, checked.key.size).unwrap(),
        [7, 8, 9]
    );
    let parent_link = dir.0.join("parent");
    symlink(&dir.0, &parent_link).unwrap();
    assert!(CheckedHef::open(&parent_link.join("model.hef")).is_ok());
}

#[test]
fn unreadable_and_missing_errors_are_path_free() {
    let dir = TempDir::new();
    let path = dir.file();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o0)).unwrap();
    for path in [&path, &dir.0.join("absent")] {
        let err = CheckedHef::open(path)
            .err()
            .expect("unreadable file accepted");
        assert!(matches!(err, WdHefError::NotFound));
        assert_path_free(&err, path);
    }
    assert_path_free(&WdHefError::Invalid, &path);
}

fn assert_path_free(err: &WdHefError, path: &Path) {
    for text in [err.to_string(), format!("{err:?}")] {
        assert!(!text.contains(path.to_str().unwrap()));
        if let Ok(home) = std::env::var("HOME") {
            assert!(!text.contains(&home));
        }
    }
}

#[test]
fn growth_after_fstat_reads_at_most_limit_plus_one() {
    let dir = TempDir::new();
    let path = dir.file();
    let mut checked = CheckedHef::open(&path).unwrap();
    assert_eq!(checked.key.size, 3);
    File::options()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&[0; 32])
        .unwrap();
    // Same reader helper as production, with an 8-byte cap to avoid a 256 MiB test allocation.
    assert!(matches!(
        read_bounded(&mut checked.file, 8, checked.key.size),
        Err(WdHefError::Invalid)
    ));
    assert_eq!(checked.file.stream_position().unwrap(), 9);
    assert_eq!(read_bounded(&b"12345678"[..], 8, 8).unwrap(), b"12345678");
}

#[test]
fn read_failure_is_path_free() {
    struct Fails;
    impl Read for Fails {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("/private/model.hef"))
        }
    }
    let err = read_bounded(Fails, 8, 0).unwrap_err();
    assert_path_free(&err, Path::new("/private/model.hef"));
}

#[cfg(hailo_stub)]
pub(crate) mod stub {
    use std::ffi::CStr;
    extern "C" {
        fn yu_hailort_stub_wd_fail(stage: i32);
        fn yu_hailort_stub_wd_count(event: usize) -> usize;
        fn yu_hailort_stub_wd_calls() -> *const std::ffi::c_char;
    }

    pub(crate) fn isolated(name: &str) -> bool {
        if std::env::var("YU_WD_T8_CHILD").as_deref() == Ok(name) {
            return false;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture"])
            .env("YU_WD_T8_CHILD", name)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        true
    }

    pub(crate) fn fail(stage: i32) {
        // SAFETY: existing stub setter; scenarios are isolated in fresh processes.
        unsafe { yu_hailort_stub_wd_fail(stage) };
    }

    pub(crate) fn counts() -> [usize; 7] {
        // SAFETY: all indices are in range of the existing stub counters.
        std::array::from_fn(|i| unsafe { yu_hailort_stub_wd_count(i) })
    }

    pub(crate) fn calls() -> String {
        // SAFETY: stub returns a live nul-terminated string, copied before the next call.
        unsafe { CStr::from_ptr(yu_hailort_stub_wd_calls()) }
            .to_str()
            .unwrap()
            .into()
    }
}

#[cfg(hailo_stub)]
use stub::*;

#[cfg(hailo_stub)]
pub(crate) fn digest(path: &Path) -> [u8; 32] {
    Sha256::digest(std::fs::read(path).unwrap()).into()
}

#[cfg(hailo_stub)]
fn load(cache: &WdHefCache, path: &Path) -> Result<(), WdHefError> {
    cache.with_model(path, digest(path), |_, _| Ok(()), |_| Ok(()))
}

#[cfg(hailo_stub)]
#[test]
fn unchanged_reuses_without_read_and_changed_inode_size_mtime_reload() {
    if isolated(
        "hailort::wd_hef::tests::unchanged_reuses_without_read_and_changed_inode_size_mtime_reload",
    ) {
        return;
    }
    let dir = TempDir::new();
    let path = dir.file();
    let cache = WdHefCache::default();
    load(&cache, &path).unwrap();
    let key = CheckedHef::open(&path).unwrap().key;
    // A cache hit must not reach metadata/configure or invoke the validator again.
    fail(1);
    BEFORE_READ.with(|hook| *hook.borrow_mut() = Some(Box::new(|| panic!("read on cache hit"))));
    cache
        .with_model(
            &path,
            digest(&path),
            |_, _| panic!("reread on cache hit"),
            |_| Ok(()),
        )
        .unwrap();
    BEFORE_READ.with(|hook| {
        hook.borrow_mut().take();
    });
    assert_eq!(counts()[1], 1);
    fail(0);
    std::fs::write(&path, [7, 8, 9, 10]).unwrap();
    let bigger = CheckedHef::open(&path).unwrap().key;
    assert_eq!(key.ino, bigger.ino);
    load(&cache, &path).unwrap();
    assert_eq!(counts()[1], 2);
    // Change only nanoseconds of mtime on the same inode and size, without sleeps.
    let file = File::options().write(true).open(&path).unwrap();
    let mtime = file.metadata().unwrap().modified().unwrap();
    file.set_times(std::fs::FileTimes::new().set_modified(mtime + Duration::from_nanos(1)))
        .unwrap();
    let newer = CheckedHef::open(&path).unwrap().key;
    assert_eq!(
        (bigger.ino, bigger.size, bigger.mtime),
        (newer.ino, newer.size, newer.mtime)
    );
    assert_ne!(bigger.mtime_nsec, newer.mtime_nsec);
    load(&cache, &path).unwrap();
    assert_eq!(counts()[1], 3);
    std::fs::rename(&path, dir.0.join("old")).unwrap();
    std::fs::write(&path, [7]).unwrap();
    load(&cache, &path).unwrap();
    assert_eq!(counts()[1], 4);
    // Every request reopens, even when the last loaded key is unchanged.
    std::fs::remove_file(&path).unwrap();
    assert!(matches!(
        cache.with_model(&path, [0; 32], |_, _| Ok(()), |_| Ok(())),
        Err(WdHefError::NotFound)
    ));
    cache.evict();
    assert_eq!(counts()[4], counts()[6]);
}

#[cfg(hailo_stub)]
#[test]
fn replacement_releases_before_open_and_holds_mutex() {
    if isolated("hailort::wd_hef::tests::replacement_releases_before_open_and_holds_mutex") {
        return;
    }
    let dir = TempDir::new();
    let path = dir.file();
    let cache = WdHefCache::default();
    load(&cache, &path).unwrap();
    std::fs::write(&path, [1]).unwrap();
    BEFORE_READ.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(|| {
            assert_eq!(counts()[4], 1, "old handle still alive before read");
            assert_eq!(calls(), "open,metadata,configure,release");
        }))
    });
    cache
        .with_model(
            &path,
            digest(&path),
            |_, _| {
                assert!(cache.slot.try_lock().is_err());
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();
    assert_eq!(
        calls(),
        "open,metadata,configure,release,open,metadata,configure"
    );
    cache.evict();
    assert!(cache.slot.lock().unwrap().is_none());
}

#[cfg(hailo_stub)]
#[test]
fn rejected_metadata_releases_without_configure() {
    if isolated("hailort::wd_hef::tests::rejected_metadata_releases_without_configure") {
        return;
    }
    let dir = TempDir::new();
    let path = dir.file();
    let cache = WdHefCache::default();
    fail(4);
    let result = cache.with_model(
        &path,
        digest(&path),
        |inputs, _| {
            assert_ne!(
                inputs[0].format_order,
                super::super::ffi::HAILO_FORMAT_ORDER_NHWC
            );
            Err(WdHefError::Invalid)
        },
        |_| Ok(()),
    );
    assert!(matches!(result, Err(WdHefError::Invalid)));
    assert_eq!(calls(), "open,metadata,release");
    assert!(cache.slot.lock().unwrap().is_none());
}

#[cfg(hailo_stub)]
#[test]
fn dd_cached_digest_mismatch_rereads_before_rejecting() {
    let dir = TempDir::new();
    let path = dir.file();
    let cache = WdHefCache::default();
    load(&cache, &path).unwrap();
    let before = READ_ATTEMPTS.with(|count| count.get());
    let opens = counts()[1];
    cache
        .with_model(&path, digest(&path), |_, _| Ok(()), |_| Ok(()))
        .unwrap();
    assert_eq!(READ_ATTEMPTS.with(|count| count.get()), before);
    let mut wrong = digest(&path);
    wrong[31] ^= 1;
    assert!(matches!(
        cache.with_model(&path, wrong, |_, _| Ok(()), |_| Ok(())),
        Err(WdHefError::Invalid)
    ));
    assert_eq!(READ_ATTEMPTS.with(|count| count.get()), before + 1);
    assert_eq!(counts()[1], opens);
}

#[cfg(hailo_stub)]
#[test]
fn replacement_failures_leave_only_wd_empty() {
    if isolated("hailort::wd_hef::tests::replacement_failures_leave_only_wd_empty") {
        return;
    }
    let dir = TempDir::new();
    let path = dir.file();
    let cache = WdHefCache::default();
    for stage in [1, 2, 3, 5] {
        fail(0);
        load(&cache, &path).unwrap();
        File::options()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0])
            .unwrap();
        fail(stage);
        let err = load(&cache, &path).unwrap_err();
        assert_path_free(&err, &path);
        assert!(cache.slot.lock().unwrap().is_none());
        assert_eq!(counts()[4], counts()[6]);
        let clip = super::super::ShimYolo::create("clip.hef").unwrap();
        drop(clip);
        assert_eq!(super::super::stub_counts().vdevice_create, 1);
    }
}
