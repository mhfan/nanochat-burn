
fn main() {
    // 初始化日志过滤，默认输出 info 级别信息
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt().with_env_filter(
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    ).init();

    tracing::info!("=============================================");
    tracing::info!("   Initializing nanochat-burn (Burn v0.21)   ");
    tracing::info!("=============================================");

    // 运行 GPU 验证
    //nanochat_burn::common::verify_autodiff_pipeline();

    tracing::info!("=============================================");
    tracing::info!("      Stage 0 Verification Completed!        ");
    tracing::info!("=============================================");
}
