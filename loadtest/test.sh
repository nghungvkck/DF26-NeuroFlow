#!/bin/bash
wrk -t4 -c200 -d60s http://localhost:8080/
