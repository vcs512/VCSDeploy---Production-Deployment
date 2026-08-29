# Computer Vision Deployment Revision

Deployment techniques for computer vision.

## Development

- Python language
- uv dependency manager
- docker compose for running services
- SOLID principles
- Concise comments (never inline)
- Google style docstrings (always state args and returns when not 'None')
- Ruff linter
- Never use functions inside functions (or classes inside classes)
- Explicit arguments names in functions (avoid positionals)
- Layered architecture (folders: `services/`, `core/`, `schemas/`)
- DTO for more than one object returned in a method (pydantic)
- Services configurations in JSON (`configs/` directory)
- Every service must have the inputs, usage and outputs expected described in
    the project README.md
- Every service should run as a docker compose service
- `data/` directory contains the dataset to be used
- `checkpoints/` directory contains the AI models
- `tests/` directory contains unit tests
- Working with reduced VRAM (4 GB)
