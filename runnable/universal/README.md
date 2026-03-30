# Universal EDES Runner

This folder contains a single universal runner that supports all current scenarios:

- cgm
- ogtt3
- ogtt4

Script:

- edes_universal_runner.jl

Run from repo root:

```bash
julia runnable/universal/edes_universal_runner.jl [flags]
```

## Flags

- -h, --help
  - Show usage and examples.

- -scenario NAME
  - Select scenario: cgm | ogtt3 | ogtt4
  - Default is cgm.

- -params CSV
  - Use predefined parameters and skip optimization.
  - CSV must match scenario parameter format.
  - Values are validated against scenario bounds.

- -json [filename]
  - Enable JSON output.
  - Optional filename (default: results_<scenario>.json).
  - Relative filenames are saved in this folder.

- -image [filename]
  - Enable PNG curve figure output.
  - Optional filename (default: fig_<scenario>_curves.png).
  - Relative filenames are saved in this folder.

## Parameter Formats

- cgm: k1,k5,k6,sigma_g,sigma_i
- ogtt3: k1,k5,k6,sigma_g,sigma_i
- ogtt4: k1,k5,k6,k8,sigma_g,sigma_i

## Example Commands

Help:

```bash
julia runnable/universal/edes_universal_runner.jl --help
```

CGM optimization with JSON and image:

```bash
julia runnable/universal/edes_universal_runner.jl -scenario cgm -json -image
```

OGTT3 with predefined params and custom JSON:

```bash
julia runnable/universal/edes_universal_runner.jl -scenario ogtt3 -params 0.01,0.05,3.0,0.5,0.3 -json ogtt3_result.json
```

OGTT4 with predefined params and custom image name:

```bash
julia runnable/universal/edes_universal_runner.jl -scenario ogtt4 -params 0.01,0.05,3.0,7.5,0.5,0.3 -image ogtt4_curves.png
```

OGTT4 optimization with both outputs:

```bash
julia runnable/universal/edes_universal_runner.jl -scenario ogtt4 -json -image
```
