wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
mkdir test
mv test2017 test/_
rm test2017.zip

wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
mkdir train
mv train2017 train/_
rm train2017.zip
