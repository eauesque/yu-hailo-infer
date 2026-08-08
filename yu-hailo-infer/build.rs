fn main() {
    println!("cargo:rustc-check-cfg=cfg(hailo_stub)");
    println!("cargo:rerun-if-changed=src/hailort/shim.h");
    println!("cargo:rerun-if-changed=src/hailort/shim.cpp");
    println!("cargo:rerun-if-changed=src/hailort/shim_stub.cpp");
    println!("cargo:rerun-if-env-changed=HAILO_INCLUDE_DIR");

    let hailo_include = std::env::var("HAILO_INCLUDE_DIR").ok();
    let header_exists = hailo_include
        .as_deref()
        .map(|d| std::path::Path::new(d).join("hailo/hailort.hpp").exists())
        .unwrap_or_else(|| {
            ["/usr/local/include", "/usr/include"]
                .iter()
                .any(|d| std::path::Path::new(d).join("hailo/hailort.hpp").exists())
        });

    if header_exists {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .file("src/hailort/shim.cpp")
            .flag_if_supported("-std=c++17");
        if let Some(dir) = hailo_include {
            build.include(dir);
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
