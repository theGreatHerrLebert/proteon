//! Shared rayon thread pool helpers with memory-aware auto-tuning.
//!
//! # `n_threads` convention
//!
//! Every public batch operation accepts an `n_threads` argument with the
//! following semantics, mirroring the conventions users expect from
//! sklearn / polars / joblib:
//!
//! | Value           | Meaning                                    |
//! |-----------------|--------------------------------------------|
//! | `None`          | Use all available cores (default)          |
//! | `Some(-1)`      | Use all available cores                    |
//! | `Some(0)`       | Use all available cores                    |
//! | `Some(n > 0)`   | Use exactly `n` threads                    |
//!
//! **Only strictly positive integers request a specific thread count.** Any
//! non-positive value means "all cores" — a deliberate choice to prevent the
//! `n_threads=0 → serial` footgun that would otherwise silently cripple
//! parallel workloads.
//!
//! # Memory-aware auto-tuning
//!
//! On high-core-count machines, naive `n_threads = num_cpus` can cause OOM
//! when each parallel task has a non-trivial working set. For example, on a
//! 120-core / 250 GB box, batch SASA with a 3 GB CellList per task needs
//! 360 GB peak — more than the available RAM.
//!
//! `auto_threads` clamps the requested thread count by `available_ram /
//! per_task_bytes` so the parallel pool fits in memory.

/// Resolve a user-provided thread count to an integer for [`build_pool`].
///
/// Returns `0` (meaning "use all CPUs") for `None`, `Some(-1)`, and `Some(0)`.
/// Any strictly positive value is used as-is.
///
/// See the module-level docs for the full `n_threads` convention.
pub fn resolve_threads(n: Option<i32>) -> usize {
    match n {
        None => 0,
        Some(n) if n <= 0 => 0,
        Some(n) => n as usize,
    }
}

/// Build a rayon thread pool with the given thread count.
/// `n_threads = 0` means use all available CPUs.
pub fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
}

/// Read available memory from /proc/meminfo on Linux.
/// Falls back to a conservative estimate (8 GB) on failure.
fn available_memory_bytes() -> usize {
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: usize = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                if kb > 0 {
                    return kb * 1024;
                }
            }
        }
    }
    8 * 1024 * 1024 * 1024 // 8 GB fallback
}

/// Auto-tune the rayon thread count based on per-task memory budget.
///
/// Returns `min(requested_or_all_cpus, available_ram / per_task_bytes)`,
/// reserving 25% of available memory as headroom for the rest of the process.
///
/// Always returns at least 1 thread.
///
/// # Arguments
/// * `requested` — User-requested thread count (None = use all)
/// * `per_task_bytes` — Estimated peak working set per parallel task
pub fn auto_threads(requested: Option<i32>, per_task_bytes: usize) -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    // Mirror resolve_threads: any non-positive value means "all cores".
    let user_requested = match requested {
        None => cpus,
        Some(n) if n <= 0 => cpus,
        Some(n) => (n as usize).min(cpus),
    };

    if per_task_bytes == 0 {
        return user_requested;
    }

    // Reserve 25% of available memory as headroom for the main thread,
    // Python objects, libc, etc. Use the remaining 75% for parallel tasks.
    let avail = available_memory_bytes();
    let budget = avail.saturating_mul(3) / 4;
    let memory_limit = (budget / per_task_bytes).max(1);

    user_requested.min(memory_limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_threads() {
        // None, -1, and 0 all mean "use all CPUs" (rayon default sentinel = 0).
        assert_eq!(resolve_threads(None), 0);
        assert_eq!(resolve_threads(Some(-1)), 0);
        assert_eq!(resolve_threads(Some(0)), 0);
        assert_eq!(resolve_threads(Some(-42)), 0);
        // Positive integers are used as-is.
        assert_eq!(resolve_threads(Some(1)), 1);
        assert_eq!(resolve_threads(Some(4)), 4);
    }

    #[test]
    fn test_auto_threads_zero_means_all_cores() {
        // Regression: Some(0) used to silently map to 1 thread (serial).
        // It should now behave the same as None and Some(-1).
        let n_none = auto_threads(None, 0);
        let n_neg1 = auto_threads(Some(-1), 0);
        let n_zero = auto_threads(Some(0), 0);
        assert_eq!(n_none, n_neg1);
        assert_eq!(n_none, n_zero);
        assert!(n_zero >= 1);
    }

    #[test]
    fn test_auto_threads_no_budget() {
        // per_task_bytes = 0 means no memory constraint
        let n = auto_threads(Some(8), 0);
        assert_eq!(n, 8);
    }

    #[test]
    fn test_auto_threads_clamped_by_memory() {
        // Pretend each task needs the entire available memory.
        // Should clamp to 1.
        let avail = available_memory_bytes();
        let n = auto_threads(Some(120), avail);
        assert!(n <= 2, "expected ≤2 threads when each task needs all RAM, got {}", n);
    }

    #[test]
    fn test_auto_threads_respects_cpus() {
        // Even with infinite memory, can't exceed CPUs.
        let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
        let n = auto_threads(Some(10000), 1);
        assert!(n <= cpus, "expected ≤{} threads, got {}", cpus, n);
    }

    #[test]
    fn test_auto_threads_minimum_one() {
        // Even pathological inputs should return at least 1.
        let n = auto_threads(Some(120), usize::MAX);
        assert!(n >= 1);
    }
}
