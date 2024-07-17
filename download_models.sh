#!/bin/bash
mkdir models -p
cd models
curl -kLSs https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip -O
unzip models.zip
rm models.zip -f
