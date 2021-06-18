#!/bin/bash

fname_zip="SPICE-1.0.zip"
dpath_spice="aac/metrics/spice"

fpath_zip="$dpath_spice/$fname_zip"

echo "[$0] Start installation of SPICE metric java code..."

if [ ! -f "$fpath_zip" ]; then
  echo "[$0] Zip file not found, downloading from https://panderson.me..."
  wget https://panderson.me/images/SPICE-1.0.zip -P $dpath_spice
fi

dpath_unzip="$dpath_spice/SPICE-1.0"
if [ ! -d "$dpath_unzip" ]; then
  echo "[$0] Unzipping file $dpath_zip..."
  unzip $fpath_zip -d $dpath_spice

  echo "[$0] Downloading Stanford models..."
  bash $dpath_unzip/get_stanford_models.sh
fi

dpath_lib="$dpath_spice/lib"
if [ ! -d "$dpath_lib" ]; then
  echo "[$0] Moving lib directory to $dpath_spice..."
  mv "$dpath_unzip/lib" $dpath_spice
fi

dpath_jar="$dpath_spice/spice-1.0.jar"
if [ ! -f "$dpath_jar" ]; then
  echo "[$0] Moving spice-1.0.jar file to $dpath_spice..."
  mv "$dpath_unzip/spice-1.0.jar" $dpath_spice
fi

echo "[$0] Done."
exit 0
