import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import altair as alt


def render_api_demo_tab():
    
    st.header("🔌 API Demo - Dự đoán qua REST API")
    
    # API configuration
    col1, col2 = st.columns([2, 1])
    with col1:
        api_url = st.text_input(
            "API Base URL",
            value="http://localhost:8000",
            help="URL của API server (mặc định: http://localhost:8000)"
        )
    with col2:
        if st.button("🔍 Kiểm tra kết nối"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API đang hoạt động")
                    health_data = response.json()
                    st.json(health_data)
                else:
                    st.error(f"❌ API lỗi: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Không thể kết nối: {str(e)}")
    
    st.divider()
    
    demo_mode = st.radio(
        "Chọn chế độ demo",
        ["📊 Dữ liệu mẫu", "📁 Upload CSV", "✍️ Nhập thủ công"],
        horizontal=True
    )
    
    df_input = None
    
    if demo_mode == "📊 Dữ liệu mẫu":
        st.subheader("Dữ liệu mẫu")
        
        sample_option = st.selectbox(
            "Chọn tập dữ liệu mẫu",
            ["5 phút gần nhất", "1 giờ gần nhất", "1 ngày gần nhất"]
        )
        
        try:
            sample_file = "data/train_5m_autoscaling.csv"
            df_full = pd.read_csv(sample_file)
            df_full['ds'] = pd.to_datetime(df_full['ds'])
            
            if sample_option == "5 phút gần nhất":
                df_input = df_full.tail(1)
            elif sample_option == "1 giờ gần nhất":
                df_input = df_full.tail(12)
            else:
                df_input = df_full.tail(288)
            
            st.info(f"📊 Đã load {len(df_input)} dòng dữ liệu")
            st.dataframe(df_input[['ds', 'y']].head(10), use_container_width=True)
            
            if len(df_input) > 10:
                st.caption(f"... và {len(df_input) - 10} dòng nữa")
                
        except Exception as e:
            st.error(f"Lỗi load dữ liệu mẫu: {str(e)}")
    
    elif demo_mode == "📁 Upload CSV":
        st.subheader("Upload file CSV")
        
        uploaded_file = st.file_uploader(
            "Chọn file CSV (phải có cột 'ds' và 'y')",
            type=['csv'],
            help="File CSV phải có 2 cột: 'ds' (timestamp) và 'y' (giá trị)"
        )
        
        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                df_input['ds'] = pd.to_datetime(df_input['ds'])
                
                st.success(f"✅ Đã load {len(df_input)} dòng dữ liệu")
                st.dataframe(df_input[['ds', 'y']].head(10), use_container_width=True)
                
                if len(df_input) > 10:
                    st.caption(f"... và {len(df_input) - 10} dòng nữa")
                    
            except Exception as e:
                st.error(f"Lỗi đọc file: {str(e)}")
    
    else:
        st.subheader("Nhập dữ liệu thủ công")
        
        num_rows = st.number_input(
            "Số dòng dữ liệu",
            min_value=1,
            max_value=50,
            value=5,
            help="Số lượng điểm dữ liệu lịch sử để nhập"
        )
        
        end_time = datetime.now().replace(second=0, microsecond=0)
        timestamps = [end_time - timedelta(minutes=5*i) for i in range(num_rows, 0, -1)]
        
        data_rows = []
        for i, ts in enumerate(timestamps):
            col1, col2 = st.columns(2)
            with col1:
                ds = st.text_input(
                    f"Timestamp {i+1}",
                    value=ts.strftime("%Y-%m-%d %H:%M:%S"),
                    key=f"ds_{i}"
                )
            with col2:
                y = st.number_input(
                    f"Giá trị {i+1}",
                    min_value=0,
                    value=100 + i*10,
                    key=f"y_{i}"
                )
            data_rows.append({"ds": ds, "y": y})
        
        if st.button("✅ Xác nhận dữ liệu"):
            df_input = pd.DataFrame(data_rows)
            df_input['ds'] = pd.to_datetime(df_input['ds'])
            st.success("Dữ liệu đã sẵn sàng!")
            st.dataframe(df_input, use_container_width=True)
    
    st.divider()
    
    if df_input is not None and len(df_input) > 0:
        st.subheader("⚙️ Cấu hình dự đoán")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox(
                "Model",
                ["xgboost", "hybrid", "lightgbm"],
                help="Loại model để dự đoán"
            )
        
        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                ["5m", "15m", "1m"],
                help="Độ phân giải thời gian"
            )
        
        with col3:
            horizon = st.number_input(
                "Số bước dự đoán",
                min_value=1,
                max_value=50,
                value=12,
                help="Số bước thời gian cần dự đoán"
            )
        
        if st.button("🚀 Gọi API dự đoán", type="primary", use_container_width=True):
            with st.spinner("⏳ Đang gọi API..."):
                try:
                    payload = {
                        "data": df_input[['ds', 'y']].to_dict('records'),
                        "horizon": horizon,
                        "model_type": model_type,
                        "timeframe": timeframe
                    }
                    
                    for row in payload['data']:
                        row['ds'] = str(row['ds'])
                    
                    response = requests.post(
                        f"{api_url}/forecast/predict",
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result['success']:
                            st.success("✅ Dự đoán thành công!")
                            
                            predictions = pd.DataFrame(result['predictions'])
                            predictions['ds'] = pd.to_datetime(predictions['ds'])
                            
                            tab1, tab2, tab3 = st.tabs(["📊 Biểu đồ", "📋 Bảng dữ liệu", "🔧 JSON Response"])
                            
                            with tab1:
                                st.subheader("Kết quả dự đoán")
                                
                                df_hist = df_input[['ds', 'y']].copy()
                                df_hist['type'] = 'Lịch sử'
                                df_hist = df_hist.rename(columns={'y': 'value'})
                                
                                df_pred = predictions[['ds', 'yhat']].copy()
                                df_pred['type'] = 'Dự đoán'
                                df_pred = df_pred.rename(columns={'yhat': 'value'})
                                
                                df_plot = pd.concat([df_hist, df_pred], ignore_index=True)
                                
                                chart = alt.Chart(df_plot).mark_line(point=True).encode(
                                    x=alt.X('ds:T', title='Thời gian'),
                                    y=alt.Y('value:Q', title='Giá trị'),
                                    color=alt.Color('type:N', 
                                                   scale=alt.Scale(
                                                       domain=['Lịch sử', 'Dự đoán'],
                                                       range=['#1f77b4', '#ff7f0e']
                                                   ),
                                                   legend=alt.Legend(title='Loại')),
                                    strokeDash=alt.StrokeDash('type:N',
                                                             scale=alt.Scale(
                                                                 domain=['Lịch sử', 'Dự đoán'],
                                                                 range=[[0], [5, 5]]
                                                             ))
                                ).properties(
                                    width=700,
                                    height=400,
                                    title=f"Dự đoán với {model_type.upper()} ({timeframe})"
                                ).interactive()
                                
                                st.altair_chart(chart, use_container_width=True)
                                
                                if 'yhat_lower' in predictions.columns and 'yhat_upper' in predictions.columns:
                                    st.caption("📊 Khoảng tin cậy được hiển thị trong bảng dữ liệu")
                            
                            with tab2:
                                st.subheader("Dữ liệu dự đoán chi tiết")
                                
                                display_df = predictions.copy()
                                if 'ds' in display_df.columns:
                                    display_df['ds'] = display_df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
                                
                                numeric_cols = display_df.select_dtypes(include=['float64']).columns
                                display_df[numeric_cols] = display_df[numeric_cols].round(2)
                                
                                st.dataframe(display_df, use_container_width=True)
                                
                                csv = display_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Tải xuống CSV",
                                    data=csv,
                                    file_name=f"predictions_{model_type}_{timeframe}.csv",
                                    mime="text/csv"
                                )
                            
                            with tab3:
                                st.subheader("Raw API Response")
                                st.json(result)
                        
                        else:
                            st.error(f"❌ Dự đoán thất bại: {result.get('message', 'Unknown error')}")
                    
                    else:
                        st.error(f"❌ API trả về lỗi: {response.status_code}")
                        try:
                            error_detail = response.json()
                            st.json(error_detail)
                        except:
                            st.text(response.text)
                
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timeout - API mất quá nhiều thời gian để phản hồi")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Không thể kết nối đến API. Hãy chắc chắn API server đang chạy!")
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
                    st.exception(e)
        
        st.divider()
        st.subheader("📊 Metrics API Demo")
        
        col1, col2 = st.columns(2)
        with col1:
            metrics_model = st.selectbox(
                "Model cho metrics",
                ["xgboost", "hybrid", "lightgbm"],
                key="metrics_model"
            )
        with col2:
            metrics_timeframe = st.selectbox(
                "Timeframe cho metrics",
                ["5m", "15m", "1m"],
                key="metrics_timeframe"
            )
        
        if st.button("📈 Lấy Metrics", use_container_width=True):
            with st.spinner("⏳ Đang tải metrics..."):
                try:
                    response = requests.get(
                        f"{api_url}/metrics/{metrics_model}/{metrics_timeframe}",
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        metrics_result = response.json()
                        
                        if metrics_result['success']:
                            st.success("✅ Metrics đã tải thành công!")
                            
                            metrics = metrics_result['metrics']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("MAE", f"{metrics['mae']:.2f}")
                            with col2:
                                st.metric("RMSE", f"{metrics['rmse']:.2f}")
                            with col3:
                                st.metric("MAPE", f"{metrics['mape']:.2f}%")
                            
                            with st.expander("🔧 Raw JSON Response"):
                                st.json(metrics_result)
                        else:
                            st.warning(f"⚠️ {metrics_result.get('message', 'Metrics not found')}")
                    else:
                        st.error(f"❌ API error: {response.status_code}")
                
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
    
    else:
        st.info("👆 Hãy chọn hoặc nhập dữ liệu ở trên để bắt đầu dự đoán")
    
    with st.expander("📖 Hướng dẫn sử dụng API"):
        st.markdown("""
        ### Các endpoint có sẵn:
        
        **1. Health Check**
        ```bash
        GET /health
        ```
        
        **2. Forward Prediction**
        ```bash
        POST /forecast/predict
        Content-Type: application/json
        
        {
            "data": [{"ds": "2023-01-01 00:00:00", "y": 100}],
            "horizon": 12,
            "model_type": "xgboost",
            "timeframe": "5m"
        }
        ```
        
        **3. Get Metrics**
        ```bash
        GET /metrics/{model_type}/{timeframe}
        ```
        
        **4. List Available Models**
        ```bash
        GET /models
        ```
        
        ### Python Example:
        ```python
        import requests
        
        # API call
        response = requests.post(
            "http://localhost:8000/forecast/predict",
            json={
                "data": [{"ds": "2023-01-01 00:00:00", "y": 100}],
                "horizon": 12,
                "model_type": "xgboost",
                "timeframe": "5m"
            }
        )
        
        result = response.json()
        print(result['predictions'])
        ```
        
        ### Xem thêm:
        - 📚 Swagger UI: http://localhost:8000/docs
        - 📄 ReDoc: http://localhost:8000/redoc
        - 📖 README_API.md cho hướng dẫn chi tiết
        """)
