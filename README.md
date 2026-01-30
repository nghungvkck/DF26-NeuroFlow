# Server Cluster Autoscaling Demo

## Run
```bash
[//]: # (tạo ra cụm server)
docker compose up --build

[//]: # (chạy file server_tracker.py để lưu lại lịch sử request của các server)
[//]: # (lịch sử request sẽ được lưu vào file server_history.json)
python server_tracker.py 

[//]: # (chay file live_plot.py) de xem do thi bieu dan cua ram va cpu khi ban request

[//]: # (chay file test.sh) gửi request đên từng server
cd loadtest
bash test.sh

[//]: # (sau khi reuqest xong, chạy file run_cost.py để tính toán)
[//]: # (ở đây em tạm thời tính tổng số tiền bằng thời gian xử dụng cpu và ram)

[//]: # (thu tu chay nhu tren a)
[//]: # (ở đây em có xử dụng nginx, để điều phối request đến các server, 
khi server này bận, nginx sẽ tự động điều chỉnh đến server đang dảnh để xử lý)



