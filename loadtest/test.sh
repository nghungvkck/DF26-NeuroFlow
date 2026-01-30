#!/bin/bash

FLAG_FILE="/home/nghung/Learn/DeepLearning/server-cluster/metrics/load.flag"

echo "START" > $FLAG_FILE
echo "▶️ Load test started"

wrk -t4 -c200 -d10s http://localhost:8080/

echo "END" > $FLAG_FILE
echo "⏹️ Load test finished"
