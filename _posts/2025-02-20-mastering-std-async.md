---
layout: post
title: "Mastering std::async in Modern C++"
description: "Understand launch policies, lifetime rules, and practical patterns for std::async so your asynchronous tasks finish predictably."
tags: [c++, concurrency, async, futures]
---

`std::async` is the standard library's shortest path to running work asynchronously and receiving the result as a `std::future`. Used well, it hides thread management, propagates exceptions, and helps structure CPU-bound pipelines. Misunderstood, it quietly serializes work or launches threads you never join. Let's walk through how to make the most of it.


## Table of Contents

- [What std::async Actually Does](#what-stdasync-actually-does)
- [Launch Policies Matter](#launch-policies-matter)
- [Lifetime and Blocking Rules](#lifetime-and-blocking-rules)
- [Exception Propagation](#exception-propagation)
- [Example: Parallel Accumulate](#example-parallel-accumulate)
- [Example: Overlapping CPU and I/O](#example-overlapping-cpu-and-io)
- [Handling Timeouts](#handling-timeouts)
- [Composition Strategies](#composition-strategies)
- [Common Pitfalls](#common-pitfalls)
- [Wrapping Up](#wrapping-up)

*You may want to review [Understanding Futures and Promises in Modern C++]({{ site.baseurl }}{% link _posts/2025-02-18-understanding-futures-promises-cpp.md %}) first, then follow up with [Composing Futures in Modern C++]({{ site.baseurl }}{% link _posts/2025-02-19-compound-futures-modern-cpp.md %}). For a deep dive into the move semantics that make futures efficient, see [Understanding Reference Types in Modern C++]({{ site.baseurl }}{% link _posts/2025-10-22-cpp-reference-types-explained.md %}).*

## What std::async Actually Does

```cpp
#include <future>

std::future<int> future = std::async(std::launch::async, [] {
    return heavy_computation();
});

// later
int answer = future.get();
```

`std::async` wraps the callable in a **packaged task** backed by a promise/future pair. When you call `get()`, you either receive the return value or the exception the callable threw. No explicit threads, no manual promise, just the result.

## Launch Policies Matter

The first template argument controls when and where work runs:

- `std::launch::async`: start immediately on a new thread.
- `std::launch::deferred`: delay execution until the first `.get()` or `.wait()`. Runs synchronously in that call.
- Default (no policy): implementation may pick either. You must not depend on one behavior.

```cpp
auto maybe_async = std::async([] { return compute(); });
auto definitely_async = std::async(std::launch::async, [] { return compute(); });
auto deferred = std::async(std::launch::deferred, [] { return compute(); });
```

Use `std::launch::async` explicitly when latency overlaps matter. Reserve `deferred` for expensive work that might not be needed.

## Lifetime and Blocking Rules

- The destructor of the returned `std::future` blocks if the task was launched with `std::launch::async` and you haven't called `get()` or `wait()`. Always consume the future before it goes out of scope.
- If the task was `std::launch::deferred`, the destructor does nothing—execution happens during `get()` instead.
- Copying is illegal. Move the future if you need to transfer ownership.

```cpp
std::future<void> fire_and_wait() {
    auto fut = std::async(std::launch::async, [] { work(); });
    // do other stuff
    fut.wait(); // ensures task finished before returning
    return fut; // UB: returning moved-from future. Instead, return after join
}
```

Avoid returning by value unless you're happy with move-semantics and the caller takes ownership.

## Exception Propagation

If the callable throws, the exception is stored and rethrown by `future.get()`:

```cpp
auto fut = std::async(std::launch::async, []() -> int {
    throw std::runtime_error("bad");
});

try {
    fut.get();
} catch (const std::runtime_error& e) {
    handle_error(e);
}
```

Because `std::async` already manages the promise, you don't need explicit try/catch in the lambda unless you want to translate errors.

## Example: Parallel Accumulate

```cpp
template <typename It>
int parallel_sum(It begin, It end) {
    auto length = std::distance(begin, end);
    if (length < 1'000) {
        return std::accumulate(begin, end, 0);
    }

    It mid = begin;
    std::advance(mid, length / 2);

    auto lower = std::async(std::launch::async, parallel_sum<It>, begin, mid);
    int upper = parallel_sum(mid, end);
    return lower.get() + upper;
}
```

- One branch recurses asynchronously.
- The other runs inline.
- The recursive `get()` ensures all child tasks complete before returning.

This pattern keeps the task tree bounded and avoids exhausting the thread implementation.

## Example: Overlapping CPU and I/O

```cpp
auto cpu_future = std::async(std::launch::async, run_simulation);
auto io_future  = std::async(std::launch::async, read_disk_snapshot);

simulate_ui();

auto state = io_future.get();   // wait for I/O
auto result = cpu_future.get(); // wait for simulation
```

Launching both tasks with `std::launch::async` overlaps CPU work with disk or networking. Be sure to call `get()` in all paths to avoid leaving background threads alive in destructors.

## Handling Timeouts

```cpp
auto fut = std::async(std::launch::async, run_rpc);
if (fut.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout) {
    cancel_rpc();
    // Future still needs to be consumed to avoid blocking destructor
    try {
        fut.get(); // likely throws or blocks until completion
    } catch (...) {
        // handle cancellation result
    }
}
```

You cannot truly cancel the underlying task with standard `std::async`, but you can structure your code so the worker checks a shared atomic flag.

## Composition Strategies

`std::async` returns `std::future`. To combine results:

- For simple chains, call `.get()` and feed results to the next `std::async`.
- For fan-out, hand the futures to `std::when_all` (C++20) or helpers from the [compound futures article]({{ site.baseurl }}{% link _posts/2025-02-19-compound-futures-modern-cpp.md %}).
- When you need to schedule follow-up work, wrap the call in a helper that launches another `std::async` after `.get()` completes.

## Common Pitfalls

- **Ignoring the return future**: letting it go out of scope blocks (async) or leaves work undone (deferred).
- **Mixing policies accidentally**: implementations may choose `deferred` by default. Specify the policy explicitly.
- **Long-running CPU loops**: saturating `std::async` with hot loops may spawn more threads than your system can handle. Consider using a thread pool or task scheduler.
- **Shared state**: `std::async` doesn't eliminate data races. Protect shared data with synchronization or confine ownership to the task scope.

## Wrapping Up

`std::async` shines when you want convenient, exception-safe task launching without writing a custom thread wrapper. Couple it with strong future handling practices—from basic promise/future contracts to the compound patterns we explored—and you get clean, maintainable concurrency for both small utilities and production code. Grow from here by experimenting with executors (`std::jthread`, C++23 continuations) and dedicated task frameworks when you need more control.
