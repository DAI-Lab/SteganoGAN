#!/usr/bin/env bash

# Div2K
# -------------------------
mkdir div2k
cd div2k

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
mkdir val
unzip -j DIV2K_valid_HR.zip -d val/_
rm DIV2K_valid_HR.zip

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
mkdir train
unzip -j DIV2K_train_HR.zip -d train/_
rm DIV2K_train_HR.zip

cd ..

# MSCOCO
# -------------------------
mkdir mscoco
cd mscoco

wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
mkdir val
mv test2017 val/_
rm test2017.zip

wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
mkdir train
mv train2017 train/_
rm train2017.zip

cd ..
