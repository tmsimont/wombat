if [ ! -e ../data/text8 ]; then
  mkdir ../data -p
  wget http://mattmahoney.net/dc/text8.zip -O ../data/text8.gz
  gzip -d ../data/text8.gz -f
fi
