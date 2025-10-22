---
layout: post
title: "Composing Futures in Modern C++"
description: "Techniques for coordinating multiple std::future values so complex asynchronous work stays predictable."
tags: [c++, concurrency, futures, async]
---

In the previous post we focused on the contract between a single promise and future. Real systems rarely stop there. Data pipelines, UI flows, and service backends routinely launch *several* asynchronous operations and need a coordinated response. This article explores how to build those compound futures—combining readiness, folding results, and handling failure as a single outcome.

*New to the basics? Start with [Understanding Futures and Promises in Modern C++]({% link _posts/2025-02-18-understanding-futures-promises-cpp.md %}) and come back when you're ready to compose them.*

## Why Compose Futures?

Fan-out work patterns introduce three questions:

- When can downstream code proceed safely?
- How do we aggregate results (or errors) from multiple producers?
- Can we cancel or short-circuit once one task succeeds?

Well-composed futures answer these without littering code with ad-hoc counters or mutexes. With C++20 you gain utilities such as `std::when_all` and `std::when_any`; before that, a few lines of promise bookkeeping accomplish the same goal.

## Modern Building Blocks

```cpp
#include <future>
#include <vector>

std::future<int> fetch_user_count();
std::future<double> fetch_revenue();
```

- `std::future<T>` remains single-consumer. Convert to `std::shared_future` when many listeners need the same value.
- `std::when_all` (C++20) combines readiness into a new future whose value is a tuple of the inputs.
- `std::when_any` resolves as soon as the first input becomes ready, providing its index and future.

These algorithms make it trivial to treat a batch of work as one unified step.

## Example: Waiting for All Results

```cpp
#include <future>
#include <tuple>

auto user_future    = std::async(std::launch::async, fetch_user_count);
auto revenue_future = std::async(std::launch::async, fetch_revenue);

#if defined(__cpp_lib_when_all) && __cpp_lib_when_all >= 201811L
auto summary_future = std::when_all(std::move(user_future), std::move(revenue_future));

auto summary = summary_future.get(); // blocks until both finish
auto users   = std::get<0>(summary).get();
auto revenue = std::get<1>(summary).get();
#else
auto summary_future = std::async(std::launch::async,
    [u = std::move(user_future), r = std::move(revenue_future)]() mutable {
        return std::make_tuple(u.get(), r.get());
    });

auto [users, revenue] = summary_future.get();
#endif
```

`std::when_all` returns a new future that becomes ready when every input finishes, even if some threw exceptions. Extract each subfuture from the tuple and call `get()` individually; exceptions rethrow at that point.

### C++17-Compatible Helper

For pre-C++20 code, you can lift the fallback into a reusable helper that unwraps each future in a single aggregation task:

```cpp
#include <future>
#include <tuple>
#include <utility>

template <typename... Futures>
auto fuse_all(Futures... futures) {
    return std::async(std::launch::async,
        [bundle = std::make_tuple(std::move(futures)...)]() mutable {
            return std::apply([](auto&... fs) {
                return std::tuple{fs.get()...};
            }, bundle);
        });
}
```

The helper preserves exception propagation (`get()` rethrows from the original producers) and keeps the combination logic in one place. Replace the fallback earlier with `auto summary_future = fuse_all(std::move(user_future), std::move(revenue_future));` if you prefer.

## Example: Reacting to the First Ready Task

```cpp
#include <future>

auto cache = std::async(std::launch::async, fetch_from_cache);
auto api   = std::async(std::launch::async, fetch_from_api);

#if defined(__cpp_lib_when_any) && __cpp_lib_when_any >= 201811L
auto first_ready = std::when_any(std::move(cache), std::move(api));
auto result = first_ready.get();
// result.index identifies which input won
auto value = std::get<result.index>(result.futures).get();
#else
auto winner_future = first_ready_of(std::move(cache), std::move(api));
auto value = winner_future.get();
#endif
```

The consumer can optionally cancel or detach the slower futures after extracting the early value. Libraries such as Folly or Boost.Fiber expose richer cancellation primitives if you need them today.

```cpp
#include <atomic>
#include <future>
#include <memory>
#include <thread>

template <typename T>
auto first_ready_of(std::future<T> left, std::future<T> right) {
    struct Shared {
        std::promise<T> promise;
        std::atomic<bool> fulfilled{false};
    };

    auto shared = std::make_shared<Shared>();
    auto result = shared->promise.get_future();

    auto launch = [shared](std::future<T> fut) mutable {
        try {
            T value = fut.get();
            if (!shared->fulfilled.exchange(true)) {
                shared->promise.set_value(std::move(value));
            }
        } catch (...) {
            if (!shared->fulfilled.exchange(true)) {
                shared->promise.set_exception(std::current_exception());
            }
        }
    };

    std::thread{launch, std::move(left)}.detach();
    std::thread{launch, std::move(right)}.detach();
    return result;
}
```

The first worker to call `set_value` wins. Later completions simply drop out because the `fulfilled` flag is already set. The sample detaches threads for brevity; in production code prefer joining or reusing a thread pool.

## Rolling Your Own Aggregate Future

When the standard algorithms are unavailable, a minimal pattern is to gather homogeneous futures into a single collection task:

```cpp
template <typename T>
std::future<std::vector<T>> when_all_ready(std::vector<std::future<T>> futures) {
    return std::async(std::launch::async,
        [futures = std::move(futures)]() mutable {
            std::vector<T> values;
            values.reserve(futures.size());
            for (auto& f : futures) {
                values.push_back(f.get());
            }
            return values;
        });
}
```

Each `get()` will block until its producer finishes, but there is no extra synchronization to maintain. If any input throws, the aggregator task rethrows at `get()`.

## Practical Guidance

- Prefer the C++20 `<future>` algorithms when available; they remove error-prone bookkeeping.
- Always handle exceptions inside aggregators so that one failing task does not deadlock others.
- Convert futures to `std::shared_future` when broadcasting the same result to multiple aggregates.
- Use timeouts (`wait_for`) around compound futures to avoid stalling the caller indefinitely.
- Consider higher-level libraries (Boost.Asio, Folly, HPX) when you need cancellation, continuations, or executors beyond what the standard offers.

## Wrapping Up

Compound futures let you express whole phases of asynchronous work as a single step. Whether you adopt `std::when_all` and `std::when_any` or craft a minimal helper, the pattern is the same: guard shared state with a promise, propagate exceptions, and let the consumer observe one future that captures the whole conversation. Bring these techniques into your codebase and the jump from single-task demos to production pipelines becomes far less daunting.
