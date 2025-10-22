---
layout: post
title: "Understanding Futures and Promises in Modern C++"
description: "A practical guide to std::future, std::promise, and the asynchronous patterns that make modern C++ concurrency safer and easier to reason about."
tags: [c++, concurrency, futures, promises]
---

Modern C++ puts powerful concurrency tools in your hands, but raw threads still make it easy to race on shared state, forget to join, or swallow exceptions. Futures and promises give you a higher-level contract: one side produces a value once, the other side waits for it exactly when needed. This post walks through the mental model, standard library types, and a couple of realistic patterns so you can apply them with confidence.

*Ready to coordinate multiple futures? Continue with [Composing Futures in Modern C++]({% link _posts/2025-02-19-compound-futures-modern-cpp.md %}).*

## Why Futures and Promises Exist

Threading APIs traditionally expose two sharp edges:

- **Synchronization**: you need explicit locks, condition variables, or atomics to coordinate shared data.
- **Lifetime**: it is on you to ensure a thread finishes before objects it touches disappear.

Futures and promises sidestep both issues by separating *execution* from *consumption*. The producer promises to deliver a value (or an exception). The consumer owns a future that blocks only when it calls `get()` or checks readiness. Because the future owns the lifetime handshake, you gain a disciplined way to move results, including errors, across threads.

## Anatomy of a Future/Promise Pair

The bare essentials look like this:

```cpp
#include <future>
#include <thread>

int heavy_calculation();

int main() {
    std::promise<int> value_promise;
    std::future<int> value_future = value_promise.get_future();

    std::thread worker([p = std::move(value_promise)]() mutable {
        try {
            int result = heavy_calculation();
            p.set_value(result);
        } catch (...) {
            p.set_exception(std::current_exception());
        }
    });

    int result = value_future.get(); // blocks until the worker sets a value or exception
    worker.join();
    return result;
}
```

Key takeaways:

- `std::promise<T>` lives with the producer. Call `set_value`, `set_exception`, or `set_value_at_thread_exit` exactly once.
- `std::future<T>` lives with the consumer and becomes ready when the producer fulfills the promise.
- Moves matter. Promises are move-only; capture them by value in lambdas via `std::move`.
- Exceptions cross threads. If the worker calls `set_exception`, the consumer sees that exception when it calls `get()`.

## Delivering Results with `std::async`

`std::async` is the quickest way to obtain a future without manually wiring a promise:

```cpp
#include <future>
#include <numeric>
#include <vector>

int main() {
    std::vector<int> data = {/* ... */};

    auto sum_future = std::async(std::launch::async, [data] {
        return std::accumulate(data.begin(), data.end(), 0);
    });

    // do other work ...

    int sum = sum_future.get();
}
```

Important flags:

- `std::launch::async` requests a new thread.
- `std::launch::deferred` defers execution until `get` / `wait`.
- The default policy may choose either; specify the launch policy when determinism matters.

Because `std::async` already wraps the callable in a packaged task backed by a promise/future pair, you get the same exception propagation rules for free.

## Sharing Results Safely

Only one consumer can call `get()` on a `std::future`. If multiple readers need the result, convert it to a `std::shared_future`:

```cpp
std::future<std::string> title_future = std::async(std::launch::async, fetch_title);
std::shared_future<std::string> shared = title_future.share();

#if defined(__cpp_lib_futures) && __cpp_lib_futures >= 202306L
auto render_ui = shared.then([](auto f) { draw(f.get()); }); // C++23 .then extension
auto log_ui    = shared.then([](auto f) { log(f.get()); });
#else
std::thread renderer([copy = shared]() { draw(copy.get()); });
std::thread logger([copy = shared]() { log(copy.get()); });
renderer.join();
logger.join();
#endif
```

In C++11/14/17, the standard library lacks `future::then`, but you can still copy `shared` into multiple threads and call `get()` from each. The point remains: `std::shared_future` lets many consumers observe one promised result without data races.

## Error Propagation and Timeouts

Futures make it straightforward to surface failures and respond to delays:

```cpp
auto future = std::async(std::launch::async, perform_rpc);

if (future.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready) {
    handle(future.get());
} else {
    cancel_rpc();             // optional cleanup hook
    throw std::runtime_error("RPC timed out");
}
```

- The producer can call `set_exception` directly or simply throw; the exception reappears at `get()`.
- `wait_for` / `wait_until` let you enforce deadlines without burning CPU on busy loops.

## Building Higher-Level Pipelines

Real applications combine several asynchronous steps. One portable approach is to chain futures manually with helper functions:

```cpp
template <typename T, typename Func>
auto then(std::future<T> f, Func cont) {
    return std::async(std::launch::async, [f = std::move(f), cont = std::move(cont)]() mutable {
        return cont(f.get());
    });
}

std::future<std::string> download_and_parse(std::string url) {
    auto raw_future = std::async(std::launch::async, [url] { return http_get(url); });
    return then(std::move(raw_future), [](auto response) {
        return parse_document(response.body);
    });
}
```

While C++20 and C++23 start introducing executors and `.then()` for `std::shared_future`, rolling small helpers like this keeps intent clear today. Libraries such as Folly, Boost, or HPX offer richer continuations, but the standard tools remain the lowest common denominator.

## Practical Guidance

- Prefer futures over naked thread handles when you only need a result.
- Always join or detach threads spawned manually; futures from `std::async` manage it automatically.
- Capture promises carefully. Using `std::move` in the lambda capture avoids dangling references.
- Avoid blocking in UI or latency-sensitive threads; `wait_for` can help implement responsive timeouts.
- Beware of implicit copies. Moving a future invalidates the source; share explicitly when you need fan-out.

## Wrapping Up

Futures and promises bring structure to cross-thread communication: one writer, one (or more via `shared_future`) reader, and automatic error transport. Once you internalize the contract, you can mix `std::async`, packaged tasks, and custom continuations to express concurrency in a way that is easier to test and reason about than raw threads. As executors mature in future standards, these primitives stay foundational, and mastering them now will prepare you for the next wave of C++ concurrency.
