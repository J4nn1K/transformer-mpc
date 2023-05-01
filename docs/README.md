# Training on a Google Cloud TPU VM
### Set up of Google Cloud Project
[Google's Documentation](https://cloud.google.com/tpu/docs/run-calculation-jax)
```
gcloud config set account your-email-account
gcloud config set project your-project-id
gcloud services enable tpu.googleapis.com
gcloud beta services identity create --service tpu.googleapis.com
gcloud compute tpus tpu-vm create tpu0 \
--zone=us-central1-b \
--accelerator-type=v3-8 \
--version=tpu-vm-base
gcloud compute tpus tpu-vm ssh tpu0 --zone us-central1-b
```
### Set up of VM
```
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

