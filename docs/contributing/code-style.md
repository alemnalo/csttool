# Code Style Guide

## Print Statement Formatting

### Overview

This document defines the unified format for print statements in csttool. As of the analysis date, the codebase contains **903 print statements** across 41 files with varying formats. This guide establishes a consistent approach for future development and gradual refactoring.

**Important**: This is a **transitional standard** before migrating to a proper logging framework. The formatting rules encode semantics (log levels, message types) into visual presentation, which is technical debt. This is acceptable as an intermediate step, but should not be considered the final architecture.

### Design Philosophy & Limitations

This guide makes several explicit trade-offs:

1. **Print vs Logging**: We use `print()` instead of `logging` as an intermediate step. This locks semantics into formatting (e.g., `✓` means success, but could be INFO or DEBUG depending on context). This is technical debt we accept for now.

2. **Binary Verbosity**: We use a single `verbose` flag, but the hierarchy implies at least three levels (always-on, verbose, debug). Level 3 is treated as DEBUG in the future logging mapping, even though it is currently gated by a single `verbose` flag. Future versions should support multiple verbosity levels.

3. **Unicode Portability**: We use Unicode symbols (✓, ✗, ⚠️) which render correctly on Linux/macOS terminals but may break alignment on Windows PowerShell or in redirected logs. If pipeline robustness is critical, consider ASCII fallbacks.

4. **Step Numbering Fragility**: `[Step N/Total]` assumes static totals known at runtime. Conditional or data-dependent steps will require careful handling. **Step counts are best-effort, not contractual guarantees.**

### General Principles

1. **Use the verbose flag**: Most output should be wrapped in `if verbose:` blocks
2. **Keep status messages minimal**: Only essential progress information should be always-on
3. **Use consistent symbols**: Standardize on Unicode symbols for status indication (see Unicode limitations above)
4. **Maintain clear hierarchy**: Use consistent indentation to show information nesting
5. **Use fixed separator width**: Always use `"=" * 60` for section separators
6. **Use f-strings for interpolation**: Use f-strings when interpolating variables; plain strings are acceptable otherwise. Never use concatenation (`+`) or `.format()`
7. **CLI boundary principle**: Print statements should live at CLI entrypoints, not deep in library code (exceptions allowed during transitional migration)

### Print Statement Hierarchy

```text
┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 0: Section Headers (NO indentation)                      │
│ - Purpose: Major pipeline stage boundaries                      │
│ - Format: Enclosed with "=" * 60 separators                     │
├─────────────────────────────────────────────────────────────────┤
│ LEVEL 1: Top-level steps (NO indentation)                       │
│ - Purpose: Sequential operation description                     │
│ - Format: [Step N/Total] description                            │
├─────────────────────────────────────────────────────────────────┤
│ LEVEL 2: Sub-steps/status (2-space indentation)                │
│ - Purpose: Operation results and status messages               │
│ - Format: `  {symbol} message`                                  │
├─────────────────────────────────────────────────────────────────┤
│ LEVEL 3: Detailed metrics (4-space indentation, verbose only)  │
│ - Purpose: Technical details and numerical metrics              │
│ - Format: {tree_symbol} metric_name: {value}                    │
└─────────────────────────────────────────────────────────────────┘
```

### Unified Status Symbols

Use these symbols consistently throughout the codebase:

| Symbol | Use Case             | Example                                    | Notes                          |
|--------|----------------------|--------------------------------------------|--------------------------------|
| `✓`    | Success / Completion | `✓ Saved registration: output.nii.gz`      | Level 2 (status)               |
| `✗`    | Error / Failure      | `✗ Failed: insufficient disk space`        | Level 2, then raise exception  |
| `⚠️`   | Warning / Fallback   | `⚠️ Skipped: no CST streamlines extracted` | May break alignment on Windows |
| `→`    | In-progress action   | `→ Computing: FA map...`                   | Level 2, ongoing operations    |
| `•`    | Info / Detail        | `• Voxels: 1,234,567`                      | Level 3 (verbose only)         |
| `◆`    | Key Metric           | `◆ Extraction rate: 85.3%`                 | Level 3 (verbose only)         |
| (none) | Summary listing      | `Left CST: 123 streamlines`                | Level 2, completion summaries  |
| `├─`   | Tree item            | `├─ FA mean: 0.456`                        | Level 3, intermediate          |
| `└─`   | Tree final item      | `└─ FA std: 0.123`                         | Level 3, last item             |

**Symbol usage constraints**:

- `◆`, `├─`, `└─`, `•` must **never** appear outside `if verbose:` blocks
- `→` indicates in-progress actions only (Level 2)
- Completion summaries use no symbol, just aligned text (Level 2)
- `⚠️` has variable width; if alignment is critical, consider `!` instead

**Deprecated symbols** (do not use in new code):

- `[OK]` / `[FAIL]` - Use ✓ / ✗ instead
- `DEBUG:` prefix - Use proper logging instead

### Format Templates

#### Section Header

Use **sparingly** for major pipeline stage boundaries only (e.g., preprocessing, tracking, extraction). Overuse creates visual clutter.

**When to use**: Major phase transitions (2-4 per full pipeline run)
**When NOT to use**: Individual functions, sub-operations, or frequent updates

```python
print("=" * 60)
print("CST EXTRACTION")
print("=" * 60)
```

Rules:

- Always use `60` for separator width (not 70, 80, etc.)
- No indentation
- Title should be UPPERCASE
- No newlines inside the header block
- Limit to 2-4 section headers per pipeline execution

#### Step Progress

Use for sequential operations:

```python
print(f"\n[Step 1/3] Registering template to subject space...")
print(f"  → Computing: affine registration...")
print(f"  ✓ Registered: {output_path}")
```

Rules:

- Start with `\n` to separate from previous output
- Use `[Step N/Total]` format
- Status messages use 2-space indentation
- Use appropriate symbols (→ for progress, ✓ for success)

#### Status Messages (Always-On)

Use sparingly for essential user-facing status:

```python
print(f"  ✓ Saved: {output_file}")
print(f"  ⚠️ Skipped: no data to process")
print(f"  ✗ Failed: {error_message}")
```

Rules:

- 2-space indentation
- Start with status symbol
- Keep message concise
- Only use for critical status information

**What qualifies as "always-on"**: Always-on prints should be limited to step headers, final success summaries, warnings, and errors. Progress details, metrics, and debugging information must be verbose-only.

**Error handling**: After printing `✗ Failed: ...`, you **must** either:

- Raise an exception immediately: `raise ValueError(error_message)`
- Return a non-zero exit code: `return 1` or `sys.exit(1)`

Never print an error and continue silently.

**Error context**: Print concise user-facing error messages. CLI entrypoints should catch exceptions and convert to clean one-line errors unless `--verbose` or `--debug` is set. Avoid printing stack traces to end users by default.

#### Verbose Output

Use for detailed progress and metrics:

```python
if verbose:
    print(f"    • Processed: {n_voxels:,} voxels")
    print(f"    • Duration: {elapsed:.2f}s")
    print(f"    ◆ Success rate: {rate:.1f}%")
```

**Tree-structured output** (for hierarchical metrics):

```python
if verbose:
    print(f"    • Diffusion metrics:")
    print(f"    ├─ FA mean: {fa_mean:.3f}")
    print(f"    ├─ FA std:  {fa_std:.3f}")
    print(f"    ├─ MD mean: {md_mean:.3f}")
    print(f"    └─ MD std:  {md_std:.3f}")
```

Rules:

- Always wrap in `if verbose:` block
- 4-space indentation (double the status messages)
- Use thousands separator for large numbers: `{value:,}`
- Use appropriate precision for floats (e.g., `.2f`, `.1f`)
- Use tree symbols for hierarchical data:
  - `├─` for intermediate items
  - `└─` for final items
- Align values when showing multiple related metrics (pad metric names with spaces)

#### Completion Messages

Use to indicate successful completion:

```python
print(f"\n✓ CST extraction complete")
print(f"  Left CST:  {left_count:,} streamlines")
print(f"  Right CST: {right_count:,} streamlines")
```

Rules:

- Start with `\n` to separate from previous output
- Use ✓ symbol for completion status line
- Summary statistics use 2-space indentation with **no symbol** (just aligned text)
- Align colons for better readability when showing multiple values

### Complete Example

```python
def process_data(input_path, output_path, verbose=False):
    """Process data with proper print formatting."""

    # Section header
    print("=" * 60)
    print("DATA PROCESSING")
    print("=" * 60)

    # Step 1
    print(f"\n[Step 1/3] Loading data...")
    if verbose:
        print(f"    • Input: {input_path}")
        print(f"    • Size: {file_size:,} bytes")

    data = load_data(input_path)
    print(f"  ✓ Loaded: {len(data):,} samples")

    # Step 2
    print(f"\n[Step 2/3] Processing data...")
    if verbose:
        print(f"    → Applying filters...")
        print(f"    → Computing metrics...")

    result = process(data)
    print(f"  ✓ Processed: {len(result):,} samples")

    if verbose:
        print(f"    ◆ Processing rate: {rate:.1f}%")

    # Step 3
    print(f"\n[Step 3/3] Saving results...")
    save(result, output_path)
    print(f"  ✓ Saved: {output_path}")

    # Completion
    print(f"\n✓ Processing complete")
    print(f"  Input samples:  {len(data):,}")
    print(f"  Output samples: {len(result):,}")
```

### Spacing Rules

1. **Before section headers**: Include `\n` or blank line
2. **Before steps**: Start with `\n[Step...]`
3. **Between status messages**: No extra spacing
4. **After completion**: End with `\n` only if more output follows
5. **Indentation**:
   - Level 0 (headers): 0 spaces
   - Level 1 (steps): 0 spaces (but use `[Step N/Total]` prefix)
   - Level 2 (status): 2 spaces
   - Level 3 (verbose): 4 spaces

### Anti-Patterns

Avoid these common mistakes:

#### ❌ Inconsistent separators

```python
print("=" * 70)  # Don't use different widths
print("- " * 30)  # Don't use different characters
```

#### ❌ Mixed indentation

```python
print("  Status message")   # Good: 2 spaces
print("    Status message") # Bad: inconsistent for same level
```

#### ❌ Verbose output without flag

```python
print(f"  Processed {i:,}/{total:,} items...")  # Should be in if verbose:
```

#### ❌ Multiple symbol styles

```python
print("[OK] Success")  # Old style
print("✓ Success")     # New style - pick one consistently
```

#### ❌ Unformatted numbers

```python
print(f"Voxels: {1234567}")    # Hard to read
print(f"Voxels: {1234567:,}")  # Better: 1,234,567
```

#### ❌ Printing inside tight loops

```python
# BAD: floods output and destroys performance
for i, item in enumerate(items):
    print(f"  Processing item {i}...")  # 10,000+ prints!

# GOOD: batch updates
for i, item in enumerate(items):
    if verbose and i % 1000 == 0:
        print(f"    • Processed {i:,} / {len(items):,} items...")
```

Even verbose prints inside tight loops can flood logs and hurt performance. Use batched updates (every N iterations) or progress bars (`tqdm`).

#### ❌ Mixing f-strings and concatenation

```python
# BAD: mixing styles
print("Processing " + filename)              # concatenation
print(f"Processing {filename}")              # f-string
print("Result: {}".format(result))           # .format()

# GOOD: use f-strings for interpolation, plain strings otherwise
print(f"Processing {filename}")              # f-string when interpolating
print("Starting process...")                 # plain string is fine
```

**Rule**: Use f-strings when interpolating variables. Plain strings are acceptable for static text. Never use concatenation (`+`) or `.format()` in print statements.

### CLI Boundary Policy

**Where print statements belong**:

- **CLI entrypoints** (`cli/commands/*.py`): Print statements are appropriate for user-facing output
- **Library code** (core modules): Should generally not print; use return values and let callers decide output
- **Exceptions during migration**: Existing prints deep in modules are acceptable during the transitional phase but should be documented and eventually removed

**Enforcement rule**: New print statements in non-CLI modules (anything outside `cli/`) are not allowed without an explicit comment explaining why. Code reviewers should reject unexplained prints in library code.

**Long-term goal**: All user-facing output originates from CLI layer. Library functions return data structures; CLI layer formats and prints them. This enables reusability (library can be imported without print spam) and testability.

### Migration Strategy

For existing code:

1. **New code**: Follow this guide strictly
2. **Bug fixes**: Update print statements in modified functions
3. **Refactoring**: Update entire modules during refactoring efforts
4. **Gradual adoption**: No need to update all 903 statements immediately
5. **Library isolation**: When refactoring, consider moving prints from library code to CLI boundaries

### Heuristic Checks

The following rules can be approximately checked with grep. These are heuristics, not precise linters—expect false positives. For robust enforcement, use an AST-based formatter and linter.

```bash
# Find separator lines (review for non-60 widths)
grep -RInE 'print\(f?"=" *\*' src/

# Find deprecated symbols
grep -RInE '\[OK\]|\[FAIL\]|DEBUG:' src/

# Find string concatenation in prints (crude heuristic)
grep -RInE 'print\(.*\+.*\)' src/

# Find print statements (to audit CLI vs library placement)
grep -RInE '^[[:space:]]*print\(' src/
```

**Note**: Grep-based checks are approximate. The most reliable enforcement is:

1. Code review
2. Python AST linter (e.g., `flake8` plugin)
3. Pre-commit hooks with proper parsers

Future work: Add pre-commit hooks with AST-based enforcement.

### Future Considerations

This style guide focuses on print statements. Future enhancements should consider:

1. **Structured logging**: Replace print with proper logging framework (e.g., `logging` module)
2. **Progress bars**: Use `tqdm` for long-running operations
3. **JSON output**: Add `--json` flag for machine-readable output
4. **Quiet mode**: Add `--quiet` flag to suppress all non-error output
5. **Log levels**: Implement DEBUG, INFO, WARNING, ERROR levels

### Migration Path to Logging

When migrating to Python's `logging` module, map the current hierarchy to log levels:

| Current Level | Symbol/Format          | Future Log Level             | Rationale                                      |
|---------------|------------------------|------------------------------|------------------------------------------------|
| Level 0       | `"=" * 60` headers     | `INFO`                       | Major phase boundaries are informational       |
| Level 1       | `[Step N/Total]`       | `INFO`                       | User-facing progress is informational          |
| Level 2       | `✓`, `✗`, `⚠️`, `→`    | `INFO` / `WARNING` / `ERROR` | `✓`/`→` = INFO, `⚠️` = WARNING, `✗` = ERROR    |
| Level 3       | `•`, `◆`, `├─`, `└─`   | `DEBUG`                      | Detailed metrics are debug-level information   |

**Key insight**: The current `verbose` flag maps to `logging.DEBUG` level. Always-on output maps to `INFO` / `WARNING` / `ERROR`.

This mapping ensures that when you replace:

```python
if verbose:
    print(f"    • Processing: {item}")
```

with:

```python
logger.debug(f"Processing: {item}")
```

The semantic meaning (debug-level detail) is preserved, even though the visual formatting changes.

### Related Documents

- [Development Setup](development-setup.md)
- [Architecture Overview](architecture.md)
