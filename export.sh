#!/usr/bin/env bash

./build.sh

docker save picai_baseline_unet_processor | gzip -c > picai_baseline_unet_processor_1.0.tar.gz
