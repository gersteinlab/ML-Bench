cd ../repos/bert

wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip
wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
wget https://storage.googleapis.com/bert_models/2019_05_30/cased_L-24_H-1024_A-16.zip

unzip uncased_L-12_H-768_A-12.zip
unzip wwm_cased_L-24_H-1024_A-16.zip
unzip multi_cased_L-12_H-768_A-12.zip
unzip cased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip
unzip uncased_L-24_H-1024_A-16.zip
unzip cased_L-24_H-1024_A-16.zip

cp -r ./uncased_L-12_H-768_A-12 ./model/uncased_L-12_H-768_A-12
cp -r ./wwm_cased_L-24_H-1024_A-16 ./model/wwm_cased_L-24_H-1024_A-16
cp -r ./multi_cased_L-12_H-768_A-12 ./model/multi_cased_L-12_H-768_A-12
cp -r ./cased_L-24_H-1024_A-16 ./model/cased_L-24_H-1024_A-16
cp -r ./wwm_uncased_L-24_H-1024_A-16 ./model/wwm_uncased_L-24_H-1024_A-16
cp -r ./uncased_L-24_H-1024_A-16 ./model/uncased_L-24_H-1024_A-16
cp -r ./cased_L-24_H-1024_A-16.zip ./model/cased_L-24_H-1024_A-16.zip

conda activate IF_DS
cd ../../utils
python download_model_weight_IF.py

conda activate OP_DS
python download_model_weight_OC.py

cp ./picture.jpg ../repos/if/data/pic.jpg
cp ./picture.jpg ../repos/if/example.jpg
cp ./picture.jpg ../repos/if/image.jpg
cp ./picture.jpg ../repos/if/rainbow_owl.png
cp ./picture.jpg ../repos/lavis/data/cls.jpg
cp ./picture.jpg ../repos/lavis/data/example.jpg
cp ./picture.jpg ../repos/lavis/data/image.jpg
cp ./picture.jpg ../repos/lavis/data/self.jpg
cp ./picture.jpg ../repos/lavis/data/wait_for_check.jpg
cp ./picture.jpg ../repos/lavis/image/example.jpg
cp ./picture.jpg ../repos/lavis/city.jpg
cp ./picture.jpg ../repos/lavis/example.jpg
cp ./picture.jpg ../repos/lavis/image.jpg
cp ./picture.jpg ../repos/lavis/my_city.jpg
cp ./picture.jpg ../repos/lavis/my.jpg
cp ./picture.jpg ../repos/lavis/path_to_image.jpg
cp ./picture.jpg ../repos/lavis/path_to_your_image.jpg
cp ./picture.jpg ../repos/lavis/picture.jpg
cp ./picture.jpg ../repos/lavis/self.jpg
cp ./picture.jpg ../repos/lavis/test.jpg
cp ./picture.jpg ../repos/lavis/city.jpg
cp ./picture.jpg ../repos/lavis/your_image.jpg
cp ./picture.jpg ../repos/open_clip/image/cat.png
cp ./picture.jpg ../repos/open_clip/usr/image/cat.jpg
cp ./picture.jpg ../repos/open_clip/usr/image/cat.png
cp ./picture.jpg ../repos/open_clip/cat.jpg
cp ./picture.jpg ../repos/open_clip/cat.png
cp ./picture.jpg ../repos/open_clip/image.jpg


cd ../repos/muzic/musicbert

wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzvf lmd_full.tar.gz
zip -r lmd_full.zip lmd_full
python -u preprocess.py
bash binarize_pretrain.sh lmd_full

