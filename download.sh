if [ ! -f glove.840B.300d.txt ]; then
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
  unzip glove.840B.300d.zip
fi

wget https://www.dropbox.com/s/9ivy6a5jeizffnr/ckpt.zip?dl=1 -O ckpt.zip
unzip ckpt.zip