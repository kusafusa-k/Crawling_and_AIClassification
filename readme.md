# クローリングとVGG16による画像分類

- download.py 
flickrから画像をダウンロード

- generate_data.py 
画像をnumpy形式に変換してnpyファイルに保存

- vgg16_transfer.py 
vgg16でモデルを作り、h5ファイルとして保存

- predict.py 
モデルによる予測 
$python predict.py car1.jpg

- cnn.py 
kerasでモデル構築
