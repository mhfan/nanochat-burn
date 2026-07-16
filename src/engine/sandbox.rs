//! Bounded execution for Python emitted by a model.
//!
//! Each invocation uses a fresh subprocess, scrubbed environment and private temporary
//! directory. Common destructive APIs are disabled and Unix memory usage is limited where
//! supported. This protects evaluation runs from accidental damage; it is not a security
//! boundary against adversarial code because Python can bypass these guards and network access
//! is not isolated.

use std::{env, fs::{self, File}, io::{self, Read, Write}, path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio}, sync::atomic::{AtomicU64, Ordering},
    thread::JoinHandle, time::{Duration, Instant},
};

const MAX_OUTPUT_BYTES: u64 = 1 << 20;
const DEFAULT_MAX_MEMORY_BYTES: u64 = 256 << 20;
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

const PYTHON_GUARD: &str = r#"
import builtins, os, shutil, subprocess, sys

with open(sys.argv[1], "rb") as source_file:
    source = source_file.read()

maximum_memory_bytes = __MAX_MEMORY_BYTES__
if maximum_memory_bytes is not None and sys.platform != "darwin":
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

builtins.exit = None
builtins.quit = None
builtins.help = None
for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
             "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
             "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
             "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
    if hasattr(os, name):
        setattr(os, name, None)
for name in ("rmtree", "move", "chown"):
    if hasattr(shutil, name):
        setattr(shutil, name, None)
subprocess.Popen = None
for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
    sys.modules[name] = None

exec(compile(source, "<llm>", "exec"), {"__name__": "__main__"})
"#;

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub error: Option<String>,
    pub timeout: bool,
    pub memory_exceeded: bool,
}

impl ExecutionResult {
    fn failure(error: impl Into<String>) -> Self {
        Self { success: false, stdout: String::new(), stderr: String::new(),
            error: Some(error.into()), timeout: false, memory_exceeded: false,
        }
    }

    fn timeout() -> Self {
        Self { timeout: true, ..Self::failure("Execution timed out (process killed)") }
    }

    fn from_output(status: ExitStatus, stdout: Vec<u8>, stderr: Vec<u8>) -> Self {
        let success = status.success();
        let (stdout, stderr) = (String::from_utf8_lossy(&stdout).into_owned(),
            String::from_utf8_lossy(&stderr).into_owned());
        let error = (!success).then(|| stderr.trim().lines().next_back()
            .unwrap_or("Process exited with non-zero status").to_string());
        let memory_exceeded = stderr.contains("MemoryError");
        Self { success, stdout, stderr, error, timeout: false, memory_exceeded }
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

struct TempDir(PathBuf);

impl TempDir {
    fn create() -> io::Result<Self> {
        let root = env::temp_dir();
        for _ in 0..32 {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!("nanochat-sandbox-{}-{id}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(io::ErrorKind::AlreadyExists,
            "failed to allocate a unique sandbox directory"))
    }

    fn path(&self) -> &Path { &self.0 }
}

impl Drop for TempDir { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

fn python3_path() -> PathBuf {
    let names: &[&str] = if cfg!(windows) { &["python3.exe", "python.exe"] } else { &["python3"] };
    if let Some(path) = env::var_os("PATH") {
        for directory in env::split_paths(&path) {
            for name in names {
                let candidate = directory.join(name);
                if candidate.is_file() { return candidate; }
            }
        }
    }
    PathBuf::from(names[0])
}

fn configure_clean_environment(command: &mut Command) {
    command.env_clear().env("PYTHONIOENCODING", "utf-8").env("OMP_NUM_THREADS", "1");
    if cfg!(windows) {
        if let Some(root) = env::var_os("SystemRoot") { command.env("SystemRoot", root); }
        if let Some(path) = env::var_os("PATH") { command.env("PATH", path); }
    } else {
        command.env("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin");
    }
}

pub fn execute_code(code: &str, timeout_secs: u64) -> ExecutionResult {
    execute_code_with_limits(code, timeout_secs, Some(DEFAULT_MAX_MEMORY_BYTES))
}

pub fn execute_code_with_limits(code: &str, timeout_secs: u64,
    maximum_memory_bytes: Option<u64>) -> ExecutionResult {
    let tmp_dir = match TempDir::create() {
        Ok(directory) => directory,
        Err(error) => return ExecutionResult::failure(
            format!("Failed to create temporary directory: {error}")),
    };
    let script_name = "candidate.py";
    let script_path = tmp_dir.path().join(script_name);
    let mut script = match File::create(&script_path) {
        Ok(file) => file,
        Err(error) => return ExecutionResult::failure(format!("Failed to create script: {error}")),
    };
    if let Err(error) = script.write_all(code.as_bytes()) {
        return ExecutionResult::failure(format!("Failed to write Python code: {error}"));
    }
    drop(script);

    let memory_limit = maximum_memory_bytes.map_or_else(|| "None".into(), |value| value.to_string());
    let guard = PYTHON_GUARD.replace("__MAX_MEMORY_BYTES__", &memory_limit);
    let mut command = Command::new(python3_path());
    configure_clean_environment(&mut command);
    let mut child = match command.args(["-I", "-B", "-c", &guard, script_name])
        .current_dir(tmp_dir.path()).stdin(Stdio::null())
        .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(child) => child,
        Err(error) => return ExecutionResult::failure(format!("Failed to spawn python3: {error}")),
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
            Ok(None) if start_time.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_pipe(stdout_reader);
                let _ = join_pipe(stderr_reader);
                return ExecutionResult::timeout();
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_pipe(stdout_reader);
                let _ = join_pipe(stderr_reader);
                return ExecutionResult::failure(format!("Failed while waiting for Python: {error}"));
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

    #[test] fn test_sandbox_error_reports_exception() {
        let result = execute_code("raise ValueError('Oops')", 5);
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("ValueError: Oops"));
    }

    #[test] fn test_sandbox_drains_large_output() {
        let result = execute_code("print('x' * 200000)", 5);
        assert!(result.success);
        assert_eq!(result.stdout.trim().len(), 200000);
    }

    #[test] fn test_sandbox_scrubs_environment_and_stdin() {
        let result = execute_code(
            "import os, sys\nprint(os.getenv('HOME'))\nprint(repr(sys.stdin.read()))", 5);
        assert!(result.success, "{}", result.stderr);
        assert_eq!(result.stdout.lines().collect::<Vec<_>>(), ["None", "''"]);
    }

    #[test] fn test_sandbox_disables_common_destructive_calls() {
        let result = execute_code(
            "import os, shutil, subprocess\nprint(os.remove, shutil.rmtree, subprocess.Popen)", 5);
        assert!(result.success, "{}", result.stderr);
        assert_eq!(result.stdout.trim(), "None None None");
    }

    #[cfg(target_os = "linux")]
    #[test] fn test_sandbox_enforces_memory_limit() {
        let result = execute_code_with_limits("bytearray(64 * 1024 * 1024)", 5, Some(32 << 20));
        assert!(!result.success);
        assert!(result.memory_exceeded, "{}", result.stderr);
    }
}
