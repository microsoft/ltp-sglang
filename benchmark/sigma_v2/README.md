## Steps:
1. Download model configurations
   ```bash
   bash scripts/prepare.sh
   ```
2. Run the benchmark
   ```bash
   bash scripts/run.sh 1 0 8
   ```
3. Change the batch sizes and sequence lengths in the scripts/bsz_seq.csv file

## Note
* Balanced gating works under the following condition:
   1. Now a fixed routing table is loaded on gpu, thus taking more memory.

## Install Dependencies for DeepEP
1. check if /dev/gdrdrv is on host
2. mount /dev/gdrdrv and /usr/src/nvidia-550.90.07 when launch the container
3. Inside container:
```sh
sudo apt install build-essential devscripts debhelper fakeroot pkg-config dkms
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy/packages
CUDA=/usr/local/cuda ./build-deb-packages.sh
sudo dpkg -i gdrdrv-dkms_<version>_<arch>.<platform>.deb
sudo dpkg -i libgdrapi_<version>_<arch>.<platform>.deb
sudo dpkg -i gdrcopy-tests_<version>_<arch>.<platform>.deb
sudo dpkg -i gdrcopy_<version>_<arch>.<platform>.deb
gdrcopy_copybw
wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
tar -xvf nvshmem_src_3.2.5-1.txz
git clone https://github.com/deepseek-ai/DeepEP.git
cd nvshmem_src && git apply ../DeepEP/third-party/nvshmem.patch
echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' >> /etc/modprobe.d/nvidia.conf
```
4. outside container 
```sh
sudo update-initramfs -u
```
5. inside container
```sh
CUDA_HOME=/usr/local/cuda \
GDRCOPY_HOME=/mnt/jiamin/gdrcopy \
NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_MPI_SUPPORT=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/mnt/jiamin/nvshmem_src 
cd build
make -j$(nproc)
make install
export NVSHMEM_DIR=/mnt/jiamin/nvshmem_src
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
nvshmem-info -a
cd DeepEP
NVSHMEM_DIR=/mnt/jiamin/nvshmem_src python setup.py install
```