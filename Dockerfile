FROM julia:1.9-bookworm

WORKDIR /app
COPY Project.toml /app/Project.toml
RUN julia --project=/app -e "import Pkg; Pkg.instantiate()"
COPY . .

ENV EDES_PROJECT_DIR=/app
ENV EDES_AUTO_PKG=0
ENV EDES_INPUT_DIR=/data
ENV EDES_OUTPUT_DIR=/data

ENTRYPOINT ["julia", "/app/edes_universal_runner.jl"]
CMD ["-scenario", "cgm"]
