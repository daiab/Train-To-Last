#!/usr/bin/env bash
root=`pwd`
echo "extract path: $root/data_extracted"
mkdir -p ./data_extracted
for file in `ls`; do
    find -name "*.rar" | xargs -i rar e {} ./data_extracted
    find -name "*.zip" | xargs -i unzip {} -d ./data_extracted
    find -name "*.tar.gz" | xargs -i tar -xzf {} -C ./data_extracted
done
