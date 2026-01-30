# Server Cluster Autoscaling Demo

## Run
```bash
-docker compose up --build

-(tạo thêm số bản sao nữa) : tạo thêm số bản sao, sao cho tổng 
tạo thêm và hiện tại = 5
docker-compose up --scale app=5 -d

- Run autoscaler:
cd autoscaler
python autoscaler.py

-load test
cd loadtest
bash test.sh




