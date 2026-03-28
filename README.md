# Distributed Vision Pipeline

A high-performance, real-time video processing pipeline designed for distributed environments. This project leverages C++ and CUDA for optimized performance, enabling scalable analysis of video streams for applications such as surveillance, autonomous vehicles, and industrial inspection.

## Features
- **Real-time Processing:** Achieves low-latency video analysis through optimized C++ and CUDA kernels.
- **Distributed Architecture:** Designed for deployment across multiple nodes, handling high volumes of video data.
- **Modular Components:** Easily integrate custom vision algorithms and models.
- **GPU Acceleration:** Utilizes NVIDIA CUDA for significant speedup in computationally intensive tasks.
- **Containerized Deployment:** Docker support for consistent and portable deployments.

## Getting Started

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker (optional, for containerized deployment)

### Building from Source

```bash
git clone https://github.com/Lare1998/distributed-vision-pipeline.git
cd distributed-vision-pipeline
mkdir build
cd build
cmake ..
make
```

### Running a Sample Pipeline

```bash
./bin/vision_pipeline --config ../config/sample_config.json
```

## Contributing

Contributions are welcome! Please refer to `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the Apache 2.0 License - see the `LICENSE` file for details.
