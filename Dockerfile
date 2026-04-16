FROM julia:1.12

WORKDIR /app

COPY Project.toml Manifest.toml /app/
RUN julia --project=/app -e "import Pkg; Pkg.instantiate()"
COPY . .

ENV EDES_PROJECT_DIR=/app
ENV EDES_AUTO_PKG=0
ENV EDES_INPUT_DIR=/data
ENV EDES_OUTPUT_DIR=/data

ENTRYPOINT ["julia", "/app/edes_universal_runner.jl"]
CMD ["-scenario", "cgm"]
