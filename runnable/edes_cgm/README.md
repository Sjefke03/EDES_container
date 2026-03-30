# EDES CGM Runner

This subdirectory contains a standalone CLI runner:

- `edes_cgm_runner.jl`

Run from repository root:

```bash
julia runnable/edes_cgm/edes_cgm_runner.jl [flags]
```

## Flags

- `-h`, `--help`
  - Show help and usage examples.

- `-params k1,k5,k6,sigma_g,sigma_i`
  - Use predefined parameters instead of optimization.
  - Must provide exactly 5 comma-separated numeric values.
  - Values are validated against model bounds.

- `-json [filename]`
  - Enable JSON output.
  - If no filename is provided, output is `results.json`.
  - Relative filenames are saved in this same directory (`runnable/edes_cgm`).

- `-image`
  - Save curve image as `fig_cgm_curves.png` in this same directory.

## Common Examples

Optimization only (no files):

```bash
julia runnable/edes_cgm/edes_cgm_runner.jl
```

Optimization + JSON:

```bash
julia runnable/edes_cgm/edes_cgm_runner.jl -json
```

Optimization + named JSON + image:

```bash
julia runnable/edes_cgm/edes_cgm_runner.jl -json run1.json -image
```

Predefined parameters + JSON + image:

```bash
julia runnable/edes_cgm/edes_cgm_runner.jl -params 0.01,0.05,3.0,0.5,0.3 -json preset.json -image
```

Show all flags from CLI:

```bash
julia runnable/edes_cgm/edes_cgm_runner.jl --help
```
