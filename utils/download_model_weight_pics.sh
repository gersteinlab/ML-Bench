# Current directory is ML-Bench

wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip -P repos/bert
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip -P repos/bert
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip -P repos/bert
wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip -P repos/bert
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip -P repos/bert
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip -P repos/bert
wget https://storage.googleapis.com/bert_models/2019_05_30/cased_L-24_H-1024_A-16.zip -P repos/bert

mkdir -p repos/bert/model

unzip repos/bert/uncased_L-12_H-768_A-12.zip -d repos/bert/model && rm repos/bert/uncased_L-12_H-768_A-12.zip
unzip repos/bert/wwm_cased_L-24_H-1024_A-16.zip -d repos/bert/model && rm repos/bert/wwm_cased_L-24_H-1024_A-16.zip
unzip repos/bert/multi_cased_L-12_H-768_A-12.zip -d repos/bert/model && rm repos/bert/multi_cased_L-12_H-768_A-12.zip
unzip repos/bert/cased_L-24_H-1024_A-16.zip -d repos/bert/model && rm repos/bert/cased_L-24_H-1024_A-16.zip
unzip repos/bert/wwm_uncased_L-24_H-1024_A-16.zip -d repos/bert/model && rm repos/bert/wwm_uncased_L-24_H-1024_A-16.zip
unzip repos/bert/uncased_L-24_H-1024_A-16.zip -d repos/bert/model && rm repos/bert/uncased_L-24_H-1024_A-16.zip
unzip repos/bert/cased_L-24_H-1024_A-16.zip -d repos/bert/model && rm repos/bert/cased_L-24_H-1024_A-16.zip

python utils/download_model_weight_IF.py
python utils/download_model_weight_OC.py

mkdir -p repos/if/data
mkdir -p repos/lavis/data
mkdir -p repos/lavis/image
mkdir -p repos/open_clip/image
mkdir -p repos/open_clip/usr/image

cp utils/picture.jpg repos/if/data/pic.jpg
cp utils/picture.jpg repos/if/example.jpg
cp utils/picture.jpg repos/if/image.jpg
cp utils/picture.jpg repos/if/rainbow_owl.png
cp utils/picture.jpg repos/lavis/data/cls.jpg
cp utils/picture.jpg repos/lavis/data/example.jpg
cp utils/picture.jpg repos/lavis/data/image.jpg
cp utils/picture.jpg repos/lavis/data/self.jpg
cp utils/picture.jpg repos/lavis/data/wait_for_check.jpg
cp utils/picture.jpg repos/lavis/image/example.jpg
cp utils/picture.jpg repos/lavis/city.jpg
cp utils/picture.jpg repos/lavis/example.jpg
cp utils/picture.jpg repos/lavis/image.jpg
cp utils/picture.jpg repos/lavis/my_city.jpg
cp utils/picture.jpg repos/lavis/my.jpg
cp utils/picture.jpg repos/lavis/path_to_image.jpg
cp utils/picture.jpg repos/lavis/path_to_your_image.jpg
cp utils/picture.jpg repos/lavis/picture.jpg
cp utils/picture.jpg repos/lavis/self.jpg
cp utils/picture.jpg repos/lavis/test.jpg
cp utils/picture.jpg repos/lavis/city.jpg
cp utils/picture.jpg repos/lavis/your_image.jpg
cp utils/picture.jpg repos/open_clip/image/cat.png
cp utils/picture.jpg repos/open_clip/usr/image/cat.jpg
cp utils/picture.jpg repos/open_clip/usr/image/cat.png
cp utils/picture.jpg repos/open_clip/cat.jpg
cp utils/picture.jpg repos/open_clip/cat.png
cp utils/picture.jpg repos/open_clip/image.jpg


wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz -P repos/muzic/musicbert
tar -xzvf repos/muzic/musicbert/lmd_full.tar.gz -C repos/muzic/musicbert
zip -r repos/muzic/musicbert/lmd_full.zip repos/muzic/musicbert/lmd_full
python -u repos/muzic/musicbert/preprocess.py
bash repos/muzic/musicbert/binarize_pretrain.sh lmd_full

