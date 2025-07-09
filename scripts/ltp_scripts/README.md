# Model Inference Support for [LTP](https://github.com/microsoft/ltp-megatron-lm)

## How to Build and Run the LTP Inference Docker Container

### Pre-requisites
- Ensure you have [Docker](https://www.docker.com/get-started) installed on your machine.

### Steps to Build and Run

1. **Build the Docker Image**
   Execute the following command to build the Docker image:
   ```bash
   ./build_image.sh <tag>
   ```

2. **Run the Docker Container**
   Use the appropriate script based on your device to run the Docker container:
   ```bash
   ./run_container.sh <model_path> <model_name> <device> <image> # where `<device>` can be `cuda` or `rocm`.
   ```

3. **Push the Docker Image if needed**
   Ensure you have the necessary permissions to push the image to the Docker registry. Use the following command:
   ```bash
   ./push_image.sh <device> <registry> <tag> # where `<device>` can be `cuda` or `rocm`, and `<tag>` is the tag you want to assign to the image whose default value is `latest`.
   ```

## Additional Notes

If you have any questions or need further assistance, please feel free to reach out.
