# triton

| API | Description |
|-----|----------|
| [jit](./jit.md) | JIT decorator - compiles a function using the Triton compiler |
| [autotune](./autotune.md) | Decorator for automatically tuning a function compiled with `triton.jit` |
| [heuristics](./heuristics.md) | Decorator for specifying how certain meta-parameter values are computed |
| [Config](./Config.md) | An object representing a possible kernel configuration that the autotuner may try |

```{toctree}
:maxdepth: 3
:hidden:

jit.md
autotune.md
heuristics.md
Config.md
```
