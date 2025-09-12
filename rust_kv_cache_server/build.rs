use std::io::Result;

fn main() -> Result<()> {
    // Compile Protobuf schema
    prost_build::compile_protos(&["kv_cache.proto"], &["."])?;
    
    Ok(())
}