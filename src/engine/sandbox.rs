
use std::{fs, path::Path, process::Command, time::{Duration, Instant}};

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub error: Option<String>,
    pub timeout: bool,
}

struct TempFileGuard<'a>(&'a Path);

impl<'a> Drop for TempFileGuard<'a> {
    fn drop(&mut self) { let _ = fs::remove_file(self.0); }
}

pub fn execute_code(code: &str, timeout_secs: u64) -> ExecutionResult {
    let tmp_dir = Path::new(".cache/tmp");
    if let Err(e) = fs::create_dir_all(tmp_dir) {
        return ExecutionResult {
            success: false,
            timeout: false,
            stdout: String::new(),
            stderr: String::new(),
            error: Some(format!("Failed to create temp directory: {}", e)),
        };
    }

    // Generate a unique file name
    let file_id = rand::random::<u32>();
    let temp_file_path = tmp_dir.join(format!("sandbox_{}.py", file_id));

    if let Err(e) = fs::write(&temp_file_path, code) {
        return ExecutionResult {
            success: false,
            timeout: false,
            stdout: String::new(),
            stderr: String::new(),
            error: Some(format!("Failed to write Python code to file: {}", e)),
        };
    }

    // Create the RAII guard to delete the file when leaving this function
    let _guard = TempFileGuard(&temp_file_path);

    let mut child = match Command::new("python3").arg(&temp_file_path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => {
            return ExecutionResult {
                success: false,
                timeout: false,
                stdout: String::new(),
                stderr: String::new(),
                error: Some(format!("Failed to spawn python3 process: {}", e)),
            };
        }
    };

    let start_time = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = match child.wait_with_output() {
                    Ok(o) => o,
                    Err(e) => {
                        return ExecutionResult {
                            success: false,
                            timeout: false,
                            stdout: String::new(),
                            stderr: String::new(),
                            error: Some(format!("Failed to read process output: {}", e)),
                        };
                    }
                };

                let stdout_str = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr_str = String::from_utf8_lossy(&output.stderr).into_owned();

                return ExecutionResult {
                    success: status.success(),
                    timeout: false,
                    stdout: stdout_str,
                    stderr: stderr_str,
                    error: if status.success() { None } else {
                        Some("Process exited with non-zero status".to_string())
                    },
                };
            }
            Ok(None) => {
                if start_time.elapsed() >= timeout {
                    let _ = child.kill();
                    return ExecutionResult {
                        success: false,
                        timeout: true,
                        stdout: String::new(),
                        stderr: String::new(),
                        error: Some("Execution timed out (process killed)".to_string()),
                    };
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                let _ = child.kill();
                return ExecutionResult {
                    success: false,
                    timeout: false,
                    stdout: String::new(),
                    stderr: String::new(),
                    error: Some(format!("Error while waiting for child process: {}", e)),
                };
            }
        }
    }
}

//#[cfg(test)] mod tests { use super::*;
    #[test] fn test_sandbox_success() {
        let result = execute_code("print('hello from sandbox')", 5);
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "hello from sandbox");
        assert!(result.error.is_none());
        assert!(!result.timeout);
    }

    #[test] fn test_sandbox_timeout() {
        let result = execute_code("import time\ntime.sleep(10)", 0);
        assert!(!result.success);
        assert!(result.timeout);
        assert!(result.error.unwrap().contains("timed out"));
    }

    #[test] fn test_sandbox_error() {
        let result = execute_code("raise ValueError('Oops')", 5);
        assert!(!result.success);
        assert!(result.stderr.contains("ValueError: Oops"));
    }
//}
