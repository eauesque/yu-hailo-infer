fn main() {
    println!("cargo:rustc-check-cfg=cfg(hailo_stub)");
    println!("cargo:rerun-if-changed=src/hailort/shim.h");
    println!("cargo:rerun-if-changed=src/hailort/shim.cpp");
    println!("cargo:rerun-if-changed=src/hailort/shim_stub.cpp");
    println!("cargo:rerun-if-env-changed=HAILO_INCLUDE_DIR");

    // /usr/local wins over /usr: a HailoRT version built from source (no
    // .deb available yet, e.g. 5.4.0) installs under /usr/local, and must be
    // the one this links against -- linking against an older .deb-packaged
    // /usr/lib version instead makes HailoRT refuse the device at runtime
    // with HAILO_INVALID_DRIVER_VERSION once the driver has been upgraded
    // (headers alone matching doesn't catch this: the library's SONAME is
    // baked in at link time). Search dirs are paired by prefix so the -L
    // always matches whichever -I we picked.
    let env_include = std::env::var("HAILO_INCLUDE_DIR").ok();
    let header_exists = env_include
        .as_deref()
        .map(|d| std::path::Path::new(d).join("hailo/hailort.hpp").exists())
        .unwrap_or_else(|| {
            ["/usr/local", "/usr"]
                .iter()
                .any(|prefix| std::path::Path::new(prefix).join("include/hailo/hailort.hpp").exists())
        });

    if header_exists {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .file("src/hailort/shim.cpp")
            .flag_if_supported("-std=c++17");
        if let Some(dir) = &env_include {
            build.include(dir);
        } else if let Some(prefix) = ["/usr/local", "/usr"]
            .iter()
            .find(|prefix| std::path::Path::new(prefix).join("include/hailo/hailort.hpp").exists())
        {
            build.include(format!("{prefix}/include"));
            println!("cargo:rustc-link-search=native={prefix}/lib");
        }
        build.compile("yu_hailort_shim");
        println!("cargo:rustc-link-lib=hailort");
    } else {
        // No HailoRT headers — compile stub so tests can run on non-Hailo hosts.
        println!("cargo:rustc-cfg=hailo_stub");
        cc::Build::new()
            .cpp(true)
            .file("src/hailort/shim_stub.cpp")
            .flag_if_supported("-std=c++17")
            .compile("yu_hailort_shim");
    }
    // stdc++ is auto-linked by MSVC; only needed for GCC/Clang targets
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
