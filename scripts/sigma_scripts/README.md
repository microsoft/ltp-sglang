# Sigma Model Support   
  
## How to Build and Run the Sigma Inference Docker Container  
   
### Pre-requisites  
- Ensure you have [Docker](https://www.docker.com/get-started) installed on your machine.  
- Download the `sigma` model files. You can find the model files [here](insert_link_to_sigma_model_files).  
   
### Steps to Build and Run  
   
1. **Build the Docker Image**
   Execute the following command to build the Docker image:  
   ```bash  
   ./image_build.sh  
   ```

2. **Run the Docker Container**
   Use the appropriate script based on your device to run the Docker container:
   ```bash
   ./run_container.sh <device> # where `<device>` can be `cuda` or `rocm`.
   ```

3. **Push the Docker Image if needed**
   Ensure you have the necessary permissions to push the image to the Docker registry (sigmainference.azurecr.io). Use the following command:
   ```bash
   ./image_push.sh <device> <tag> # where `<device>` can be `cuda` or `rocm`, and `<tag>` is the tag you want to assign to the image whose default value is `latest`.
   ```

## Additional Notes

If you have any questions or need further assistance, please feel free to reach out.
