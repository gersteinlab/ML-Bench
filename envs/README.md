# Environment Setup

## ML-Agent-Bench Docker Setup

To run the ML-Agent-Bench Docker container, you can use the following command:

```bash
docker pull public.ecr.aws/i5g0m1f6/ml-bench
docker run -it public.ecr.aws/i5g0m1f6/ml-bench /bin/bash
```

This will pull the latest ML-Agent-Bench Docker image and run it in an interactive shell. The container includes all the necessary dependencies to run the ML-Agent-Bench codebase.

For ML-Agent-Bench in OpenDevin, please refer to the [OpenDevin setup guide](https://github.com/OpenDevin/OpenDevin/blob/main/evaluation/ml_bench/README.md).