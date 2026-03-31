# Universal EDES Runner

This folder contains a single universal runner that supports all current scenarios:

- cgm
- ogtt3
- ogtt4

Script:

- edes_universal_runner.jl

Run from this folder (`runnable`):

```bash
julia edes_universal_runner.jl [flags]
```

## Docker One-Shot Usage

Build image from this folder (`runnable`):

```bash
docker build -t edes-universal:latest .
```

Run as one-shot container with mounted input/output folder:

```bash
docker run --rm -v "${PWD}:/io" edes-universal:latest -scenario cgm -data /io/test_data_cgm.json -json cgm_out.json -image cgm_out.png
```

Notes:

- The container entrypoint is the universal runner script.
- `/io` is the working directory in the container.
- Relative `-json` and `-image` outputs are written to `/io` (mounted host folder).
- You can also pass absolute output paths.

## Flags

- -h, --help
  - Show usage and examples.

- -scenario NAME
  - Select scenario: cgm | ogtt3 | ogtt4
  - Default is cgm.

- -data FILE
  - Load custom data from JSON file instead of using default scenario data.
  - JSON must contain "time", "glucose" arrays.
  - "insulin" array is optional for cgm (uses fasting_insulin if not provided).
  - "insulin" array is required for ogtt3 and ogtt4.
  - All values must be finite numbers.

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

## Parameter Fitting with OGTT Data

When **no `-params` flag** is provided, the script automatically runs **parameter optimization** against the measurement data:

### How Parameter Optimization Works

**Optimization Algorithm**: Nelder-Mead simplex method (gradient-free)
- Uses 300 random initial parameter guesses distributed via Latin Hypercube Sampling
- Each guess generates a separate optimization run
- Selects the best parameters from successful runs
- Typical success rate: 50-70% of runs converge to valid solutions

**OGTT3 and OGTT4 scenarios** fit parameters to both glucose AND insulin measurements:
- Script loads your OGTT measurement data (glucose points + insulin points at sample times)
- Runs optimization using multiple initial guesses  
- Each optimization run simulates the ODE model and compares predictions to your measured data
- Computes loss based on both glucose and insulin residuals
- Returns parameters that best fit your experimental measurements

**CGM scenario** fits to glucose measurements only (insulin is constant fasting level)

### Example: OGTT3 Parameter Fitting

```bash
# Fit parameters to your OGTT3 patient data
julia edes_universal_runner.jl -scenario ogtt3 -data patient_data.json -json fit_results.json -image fit_curves.png
```

Output shows:
- Measurement data plotted as red dots
- Simulated curves (blue line) fitted to your data
- JSON file with fitted parameter values
- Optimization statistics (successful runs, best objective value)

### Performance Notes
- OGTT optimization typically takes 2-5 minutes
- CGM optimization may take longer due to more measurement points
- Reduce `n_initial_guesses` value in code if speed is critical

## Example Commands

Help:

```bash
julia edes_universal_runner.jl --help
```

CGM optimization with JSON and image:

```bash
julia edes_universal_runner.jl -scenario cgm -json -image
```

OGTT3 with predefined params and custom JSON:

```bash
julia edes_universal_runner.jl -scenario ogtt3 -params 0.01,0.05,3.0,0.5,0.3 -json ogtt3_result.json
```

OGTT4 with predefined params and custom image name:

```bash
julia edes_universal_runner.jl -scenario ogtt4 -params 0.01,0.05,3.0,7.5,0.5,0.3 -image ogtt4_curves.png
```

CGM with custom data from JSON and predefined params:

```bash
julia edes_universal_runner.jl -scenario cgm -data mydata.json -params 0.01,0.05,3.0,0.5,0.3 -json -image
```

OGTT3 with custom data and optimization:

```bash
julia edes_universal_runner.jl -scenario ogtt3 -data patient_ogtt3.json -json ogtt3_fit.json
```

## Test Commands by Scenario

Run these from this folder (`runnable`).

### CGM

Without predefined parameters (optimize from test data):

```bash
julia edes_universal_runner.jl -scenario cgm -data test_data_cgm.json -json cgm_out.json -image cgm_out.png
```

With predefined parameters (from cgm_out.json):

```bash
julia edes_universal_runner.jl -scenario cgm -data test_data_cgm.json -params 0.00880779075272941,0.00955640649315587,5.0,0.0206730458249846,25.04999999999984 -json cgm_predef_out.json -image cgm_predef_out.png
```

### OGTT3

Without predefined parameters (optimize from test data):

```bash
julia edes_universal_runner.jl -scenario ogtt3 -data test_data_ogtt3.json -json ogtt3_out.json -image ogtt3_out.png
```

With predefined parameters (from ogtt3_out.json):

```bash
julia edes_universal_runner.jl -scenario ogtt3 -data test_data_ogtt3.json -params 0.015267477934545208,0.050921296545794935,1.5531234642813734,0.010000000000000002,0.20032085650634182 -json ogtt3_predef_out.json -image ogtt3_predef_out.png
```

### OGTT4

Without predefined parameters (optimize from test data):

```bash
julia edes_universal_runner.jl -scenario ogtt4 -data test_data_ogtt4.json -json ogtt4_out.json -image ogtt4_out.png
```

With predefined parameters (from ogtt4_out.json):

```bash
julia edes_universal_runner.jl -scenario ogtt4 -data test_data_ogtt4.json -params 0.014518160037517042,0.05188548641056432,1.5308380035708968,5.5327535782440815,0.010000000000000002,0.1697258921293979 -json ogtt4_predef_out.json -image ogtt4_predef_out.png
```

## JSON Data Format

Custom data files should follow this structure:

```json
{
  "scenario": "cgm",
  "time": [0, 5, 10, 15, 20, 25,...],
  "glucose": [6.1, 6.2, 6.4, 6.7, 7.1, 7.6,...],
  "insulin": [7.9, 7.9, 7.9, ...],
  "fasting_insulin": 7.9
}
```

- `scenario`: Optional. Can be used to auto-detect scenario type.
- `time`: Required. Time points in minutes.
- `glucose`: Required. Glucose measurements in mmol/L.
- `insulin`: Optional for CGM, required for OGTT3/OGTT4. Insulin in mU/L.
- `fasting_insulin`: Optional. Used for CGM if insulin array not provided (default: 7.9).

OGTT4 optimization with both outputs:

```bash
julia edes_universal_runner.jl -scenario ogtt4 -json -image
```
