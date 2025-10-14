use std::io::Result;

fn main() -> Result<()> {
    // Compile protobuf definitions
    tonic_build::configure()
        .build_server(true)
        .build_client(false) // We only need server side for now
        .compile_protos(&["proto/kv_cache.proto"], &["proto"])?;

    // Rerun build if proto file changes
    println!("cargo:rerun-if-changed=proto/kv_cache.proto");

    Ok(())
}
