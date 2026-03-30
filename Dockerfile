FROM julia:1.9-bookworm

WORKDIR /app

COPY runnable/universal/Project.toml /app/runnable/universal/Project.toml
RUN julia --project=/app/runnable/universal -e "import Pkg; Pkg.instantiate()"

COPY . .

# Default working dir for mounted input/output files.
WORKDIR /io
ENV EDES_OUTPUT_DIR=/io
ENV EDES_PROJECT_DIR=/app/runnable/universal

ENTRYPOINT ["julia", "/app/runnable/universal/edes_universal_runner.jl"]
CMD ["--help"]
