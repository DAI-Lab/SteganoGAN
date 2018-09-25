# Caltech256
wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -xvf 256_ObjectCategories.tar
rm 256_ObjectCategories.tar
mv 256_ObjectCategories caltech256

# COCO
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
mkdir mscoco
mv train2017 mscoco/
