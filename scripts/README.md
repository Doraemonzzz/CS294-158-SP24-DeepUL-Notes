## Set up file paths and data
This part is direct copy from [NVAE](https://github.com/NVlabs/NVAE). For large datasets, we store the data in LMDB datasets
for I/O efficiency. Click below on each dataset to see how you can prepare your data. Below, `$DATA_DIR` indicates
the path to a data directory that will contain all the datasets and `$CODE_DIR` refers to the code directory:

<details><summary>MNIST and CIFAR-10</summary>

These datasets will be downloaded automatically, when you run the main training for NVAE using `train.py`
for the first time. You can use `--data=$DATA_DIR/mnist` or `--data=$DATA_DIR/cifar10`, so that the datasets
are downloaded to the corresponding directories.
</details>

<details><summary>CelebA 64</summary>
Run the following commands to download the CelebA images and store them in an LMDB dataset:

```shell script
cd $CODE_DIR/scripts
python create_celeba64_lmdb.py --split train --img_path $DATA_DIR/celeba_org --lmdb_path $DATA_DIR/celeba64_lmdb
python create_celeba64_lmdb.py --split valid --img_path $DATA_DIR/celeba_org --lmdb_path $DATA_DIR/celeba64_lmdb
python create_celeba64_lmdb.py --split test  --img_path $DATA_DIR/celeba_org --lmdb_path $DATA_DIR/celeba64_lmdb
```
Above, the images will be downloaded to `$DATA_DIR/celeba_org` automatically and then then LMDB datasets are created
at `$DATA_DIR/celeba64_lmdb`.
</details>

<details><summary>ImageNet 32x32</summary>

Run the following commands to download tfrecord files from [GLOW](https://github.com/openai/glow) and to convert them
to LMDB datasets
```shell script
mkdir -p $DATA_DIR/imagenet-oord
cd $DATA_DIR/imagenet-oord
wget https://storage.googleapis.com/glow-demo/data/imagenet-oord-tfr.tar
tar -xvf imagenet-oord-tfr.tar
cd $CODE_DIR/scripts
python convert_tfrecord_to_lmdb.py --dataset=imagenet-oord_32 --tfr_path=$DATA_DIR/imagenet-oord/mnt/host/imagenet-oord-tfr --lmdb_path=$DATA_DIR/imagenet-oord/imagenet-oord-lmdb_32 --split=train
python convert_tfrecord_to_lmdb.py --dataset=imagenet-oord_32 --tfr_path=$DATA_DIR/imagenet-oord/mnt/host/imagenet-oord-tfr --lmdb_path=$DATA_DIR/imagenet-oord/imagenet-oord-lmdb_32 --split=validation
```
</details>

<details><summary>CelebA HQ 256</summary>

Run the following commands to download tfrecord files from [GLOW](https://github.com/openai/glow) and to convert them
to LMDB datasets
```shell script
mkdir -p $DATA_DIR/celeba
cd $DATA_DIR/celeba
wget https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar
tar -xvf celeba-tfr.tar
cd $CODE_DIR/scripts
python convert_tfrecord_to_lmdb.py --dataset=celeba --tfr_path=$DATA_DIR/celeba/celeba-tfr --lmdb_path=$DATA_DIR/celeba/celeba-lmdb --split=train
python convert_tfrecord_to_lmdb.py --dataset=celeba --tfr_path=$DATA_DIR/celeba/celeba-tfr --lmdb_path=$DATA_DIR/celeba/celeba-lmdb --split=validation
```
</details>


<details><summary>FFHQ 256</summary>

Visit [this Google drive location](https://drive.google.com/drive/folders/1WocxvZ4GEZ1DI8dOz30aSj2zT6pkATYS) and download
`images1024x1024.zip`. Run the following commands to unzip the images and to store them in LMDB datasets:
```shell script
mkdir -p $DATA_DIR/ffhq
unzip images1024x1024.zip -d $DATA_DIR/ffhq/
cd $CODE_DIR/scripts
python create_ffhq_lmdb.py --ffhq_img_path=$DATA_DIR/ffhq/images1024x1024/ --ffhq_lmdb_path=$DATA_DIR/ffhq/ffhq-lmdb --split=train
python create_ffhq_lmdb.py --ffhq_img_path=$DATA_DIR/ffhq/images1024x1024/ --ffhq_lmdb_path=$DATA_DIR/ffhq/ffhq-lmdb --split=validation
```
</details>

<details><summary>LSUN</summary>

We use LSUN datasets in our follow-up works. Visit [LSUN](https://www.yf.io/p/lsun) for
instructions on how to download this dataset. Since the LSUN scene datasets come in the
LMDB format, they are ready to be loaded using torchvision data loaders.

</details>
