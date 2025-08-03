# Sigma README

## Prerequisites

1. To run the Sigma model with SGLang, you can refer to README files in the `sglang/scripts/ltp_scripts` directory to build and run the Docker container.

2. Ensure you have download the Sigma model weights.
    - Please CHECK that the `config.json` file in the model weight directory: the `architectures` field should be set to `["SigmaV2ForCausalLM"]`. In some initial versions of model weights, it may be set to `["SigmaForCausalLM"]`, which is not compatible with the current SGLang implementation, and it will be updated in the future.

3. In the docker container, you can use the following command to run the Sigma model with SGLang:

   ```bash
   ./run-sglang-sigma.sh <model_path> <tp_size> <enable_ep> <attention_backend>
   ```

    - `<model_path>`: Path to the Sigma model weights.
    - `<tp_size>`: Tensor parallel size, e.g., `8`.
    - `<enable_ep>`: Whether to enable expert parallelism, e.g., `true` or `false`.
    - `<attention_backend>`: Attention backend to use. `triton` is the default. For nvidia GPUs, you can also use `fa3` which will be better.

## Additional Notes
If you have any questions or need further assistance, please feel free to reach out.
