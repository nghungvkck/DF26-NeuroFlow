#!/bin/bash
wrk -t4 -c200 -d10s http://localhost:8080/


#dung wrk thuc hien request trong 10s
