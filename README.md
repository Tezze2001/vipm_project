# vipm_project
Project of Visual Information Processing and Management exam

# Links for installing Faiss
[faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

[CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

# Dataset

[small_dataset.zip](https://drive.google.com/file/d/1qSCuUL8C923zxXsUkGyIxxpG4HLXmWNe/view?usp=sharing)


# Conda setup (non so se funziona)
Ho esportato l'ambiente conda che sto usando con:
  ```bash
  conda env export --no-builds > environment.yml
  ```
Dovrebbe contenere i pacchetti giusti per usa CUDA

Teoricamente per importarlo dovrebbe bastare fare
```bash
conda env create -f environment.yml
```