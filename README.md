## Build Presets

The project supports multiple CMake build presets, configured in `CMakePresets.json`.

| Preset             | `-O3` | `-g` | `-G` (CUDA) | `-pg` | `-fno-omit-frame-pointer` | 🧠 Use Case                         |
|--------------------|:-----:|:----:|:-----------:|:-----:|:--------------------------:|------------------------------------|
| 🟢 `debug`          | ❌    | ✅   | ✅          | ❌    | ✅                          | Full source-level debugging (host + CUDA) |
| 🔴 `release`        | ✅    | ❌   | ❌          | ❌    | ❌                          | Optimized build with fast math      |
| 🟡 `release-profiling` | ✅ | ❌   | ❌          | ✅    | ❌                          | Instrumented build for `gprof`      |
| 🟠 `perf-profiling` | ✅    | ✅   | ❌          | ❌    | ✅                          | Linux `perf`, flamegraphs           |

Legend:
- ✅ = Enabled
- ❌ = Disabled
- `-G`: CUDA device code debugging
- `-pg`: Profiling with `gprof`
- `-fno-omit-frame-pointer`: Required for good stack traces in `perf`

