wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
mkdir test
unzip -j DIV2K_valid_HR.zip -d test/_

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
mkdir train
unzip -j DIV2K_train_HR.zip -d train/_

rm *.zip
