# Repository Guidelines

## Project Structure & Module Organization

This is a Rust library crate for incremental HDBSCAN clustering. Core code lives in `src/`, with `src/lib.rs` exposing the public API and modules such as `hdbscan.rs`, `bubble_tree.rs`, `data_bubble.rs`, `cf.rs`, `distance.rs`, and `types.rs`. Integration tests live in `tests/` and use `*_test.rs` filenames. Project docs are in `README.md`, `ARCHITECTURE.md`, and `SPEC.md`. Build artifacts in `target/` are generated and should not be edited.

## Build, Test, and Development Commands

Prefix shell commands with `rtk` in this workspace.

```bash
rtk cargo test
rtk cargo test --features turbovec
rtk cargo check
rtk cargo fmt --check
rtk cargo clippy --all-targets --all-features
```

`cargo test` runs unit and integration tests. The `turbovec` feature enables the optional approximate k-NN dependency. `cargo check` validates compilation quickly. Use `cargo fmt --check` and `cargo clippy` before submitting changes.

## Coding Style & Naming Conventions

Use idiomatic Rust 2021 and rustfmt defaults. Prefer four-space indentation, `snake_case` for functions, modules, variables, and test names, and `UpperCamelCase` for structs, enums, and traits. Keep public API exports centralized in `src/lib.rs`. Return `Result<_, HdbscanError>` for fallible library operations instead of panicking, except in tests.

## Testing Guidelines

Place focused unit tests near implementation modules with `#[cfg(test)]`; place cross-module behavior in `tests/`. Name tests by expected behavior, for example `test_hdbscan_detects_two_bubble_clusters`. Cover insertion, deletion, clustering labels, dimension validation, and feature-gated behavior when touching those areas. Run both default tests and `--features turbovec` when changes affect neighbor search or clustering internals.

## Commit & Pull Request Guidelines

Recent history uses short Conventional Commit-style prefixes such as `feat:`, `fix:`, and `docs:`. Keep commits scoped and imperative, for example `fix: handle empty bubble tree after deletion`. Pull requests should include a clear summary, relevant test results, linked issues when applicable, and notes on algorithmic or performance tradeoffs. Include before/after metrics or sample outputs for changes that affect clustering behavior.

## Agent-Specific Instructions

Follow the repository instruction to run commands through `rtk`. Do not revert unrelated work in the tree. When editing algorithmic code, check `ARCHITECTURE.md` and `SPEC.md` first so changes remain aligned with the Bubble-tree and HDBSCAN design.
