FROM julia:1.9-bookworm

WORKDIR /app

COPY Project.toml /app/Project.toml
RUN julia --project=/app -e "import Pkg; Pkg.instantiate()"

COPY . .

# Default working dir for mounted input/output files.
WORKDIR /io
ENV EDES_OUTPUT_DIR=/io
ENV EDES_PROJECT_DIR=/app
ENV EDES_AUTO_PKG=0

ENTRYPOINT ["julia", "/app/edes_universal_runner.jl"]
CMD ["--help"]
