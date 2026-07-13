
use std::{fs::{self, File, OpenOptions}, io::{self, Read, Write}, path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio}, thread::JoinHandle, time::{Duration, Instant},
};

const MAX_OUTPUT_BYTES: u64 = 1 << 20;

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub error: Option<String>,
    pub timeout: bool,
}

impl ExecutionResult {
    fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            stdout: String::new(),
            stderr: String::new(),
            error: Some(error.into()),
            timeout: false,
        }
    }

    fn timeout() -> Self {
        Self { timeout: true, ..Self::failure("Execution timed out (process killed)") }
    }

    fn from_output(status: ExitStatus, stdout: Vec<u8>, stderr: Vec<u8>) -> Self {
        let success = status.success();
        Self { success, timeout: false,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
            error: (!success).then(|| "Process exited with non-zero status".to_string()),
        }
    }
}

fn read_pipe(mut pipe: impl Read + Send + 'static) -> JoinHandle<io::Result<Vec<u8>>> {
    std::thread::spawn(move || {
        let mut output = Vec::new();
        pipe.by_ref().take(MAX_OUTPUT_BYTES).read_to_end(&mut output)?;
        Ok(output)
    })
}

fn join_pipe(reader: JoinHandle<io::Result<Vec<u8>>>) -> Result<Vec<u8>, String> {
    reader.join().map_err(|_| "Output reader thread panicked".to_string())?
        .map_err(|error| format!("Failed to read process output: {error}"))
}

struct TempFileGuard(PathBuf);

impl Drop for TempFileGuard { fn drop(&mut self) { let _ = fs::remove_file(&self.0); } }

fn create_temp_script(directory: &Path) -> io::Result<(PathBuf, File)> {
    for _ in 0..16 {
        let path = directory.join(format!("sandbox_{}.py", rand::random::<u64>()));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(io::ErrorKind::AlreadyExists,
        "failed to allocate a unique sandbox script name"))
}

pub fn execute_code(code: &str, timeout_secs: u64) -> ExecutionResult {
    let tmp_dir = Path::new(".cache/tmp");
    if let Err(e) = fs::create_dir_all(tmp_dir) {
        return ExecutionResult::failure(format!("Failed to create temp directory: {}", e));
    }

    let (temp_file_path, mut temp_file) = match create_temp_script(tmp_dir) {
        Ok(result) => result,
        Err(e) => return ExecutionResult::failure(format!("Failed to create script: {e}")),
    };
    let _guard = TempFileGuard(temp_file_path.clone());
    if let Err(e) = temp_file.write_all(code.as_bytes()) {
        return ExecutionResult::failure(format!("Failed to write Python code to file: {}", e));
    }
    drop(temp_file);

    let script_name = temp_file_path.file_name().expect("temporary script must have a file name");
    let mut child = match Command::new("python3").args(["-I", "-B"]).arg(script_name)
        .current_dir(tmp_dir)
        .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => {
            return ExecutionResult::failure(format!("Failed to spawn python3 process: {}", e));
        }
    };
    let stdout_reader = read_pipe(child.stdout.take().expect("stdout pipe must be available"));
    let stderr_reader = read_pipe(child.stderr.take().expect("stderr pipe must be available"));

    let (start_time, timeout) = (Instant::now(), Duration::from_secs(timeout_secs));

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = match join_pipe(stdout_reader) {
                    Ok(output) => output,
                    Err(error) => return ExecutionResult::failure(error),
                };
                let stderr = match join_pipe(stderr_reader) {
                    Ok(output) => output,
                    Err(error) => return ExecutionResult::failure(error),
                };
                return ExecutionResult::from_output(status, stdout, stderr);
            }
            Ok(None) => {
                if start_time.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = join_pipe(stdout_reader);
                    let _ = join_pipe(stderr_reader);
                    return ExecutionResult::timeout();
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_pipe(stdout_reader);
                let _ = join_pipe(stderr_reader);
                return ExecutionResult::failure(format!(
                    "Error while waiting for child process: {}", e
                ));
            }
        }
    }
}

#[cfg(test)] mod tests { use super::*;
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

    #[test] fn test_sandbox_drains_large_output() {
        let result = execute_code("print('x' * 200000)", 5);
        assert!(result.success);
        assert_eq!(result.stdout.trim().len(), 200000);
    }
}
