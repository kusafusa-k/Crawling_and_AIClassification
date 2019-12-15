# クローリングとVGG16による画像分類

- download.py <br>
flickrから画像をダウンロード

- generate_data.py <br>
画像をnumpy形式に変換してnpyファイルに保存

- vgg16_transfer.py <br>
vgg16でモデルを作り、h5ファイルとして保存

- predict.py <br>
モデルによる予測 <br>
$python predict.py car1.jpg
