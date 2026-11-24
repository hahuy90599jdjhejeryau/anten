# LOOPLAB — Web mô phỏng anten khung tròn nhỏ

Ứng dụng Streamlit chuyển phần mềm Tkinter mô phỏng anten khung tròn nhỏ sang web responsive, có thể đưa lên GitHub và triển khai bằng Streamlit Community Cloud.

## Chức năng

- Nhập bán kính vòng `R`, tần số `f`, bán kính dây `a`, số vòng `N`, trở kháng tham chiếu `Z0`.
- Chọn vật liệu dây: đồng, bạc, vàng, nhôm.
- Tính gần đúng `Rr`, `L`, `Rdc`, `Rac`, `Zin`, hiệu suất, Gain, Return Loss, SWR và realized gain.
- Hiển thị mô hình hình học anten 3D có thể xoay/phóng to.
- Hiển thị giản đồ bức xạ 3D và polar 2D.
- Khảo sát theo bán kính, tần số và bản đồ nhiệt R–f.
- So sánh vật liệu ở cùng hình học: điện trở suất, tổn hao, hiệu suất, Gain, trở kháng và khối lượng dây ước tính.
- Sinh dataset tổng hợp, huấn luyện Random Forest hoặc ANN, đánh giá MAE/RMSE/R², dự đoán bộ tham số hiện tại.
- Tải CSV kết quả, dữ liệu quét, dataset AI và bảng metric.

## Cấu trúc thư mục

```text
anten_loop_web_project/
├── app.py                 # Giao diện Streamlit
├── antenna_core.py        # Công thức và mô hình anten
├── ai_engine.py           # Sinh dataset, huấn luyện và dự đoán ML
├── requirements.txt       # Thư viện cần cài
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Chạy thử trên máy tính

Yêu cầu Python 3.11 trở lên; khuyến nghị Python 3.12 để dễ đồng nhất môi trường local và Streamlit Cloud. Project đang khóa Streamlit ở phiên bản 1.67.0.

### Windows PowerShell

```powershell
cd antenna_loop_web_project
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
cd antenna_loop_web_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Trên Windows, có thể nhấp đúp `run_windows.bat` để tạo môi trường ảo, cài thư viện và mở web. Trên macOS/Linux có thể chạy `./run_linux.sh`. Sau khi chạy, Streamlit hiển thị địa chỉ local, thường là `http://localhost:8501`.

## Đưa lên GitHub và Streamlit Cloud

1. Tạo repository mới trên GitHub.
2. Upload toàn bộ file/thư mục trong project; không upload `.venv`.
3. Vào <https://share.streamlit.io/> và đăng nhập bằng GitHub.
4. Chọn **Create app** hoặc **New app**.
5. Chọn repository, branch và đặt **Main file path** là `app.py`.
6. Nhấn **Deploy**. Streamlit Cloud sẽ đọc `requirements.txt` và cài dependencies.
7. Sau khi deploy thành công, mở URL ứng dụng trên máy tính hoặc điện thoại.

Nếu cập nhật code, commit/push lên branch đã liên kết; Streamlit Cloud sẽ tự build lại.

## Về mô hình vật liệu

Giá trị điện trở suất được dùng là giá trị khối gần đúng quanh 20 °C:

| Vật liệu | Điện trở suất (Ω·m) | Mật độ (kg/m³) |
|---|---:|---:|
| Bạc (Ag) | 1.59 × 10⁻⁸ | 10,490 |
| Đồng (Cu) | 1.724 × 10⁻⁸ | 8,960 |
| Vàng (Au) | 2.44 × 10⁻⁸ | 19,320 |
| Nhôm (Al) | 2.82 × 10⁻⁸ | 2,700 |

Trong mô hình hiện tại, khi chỉ đổi vật liệu thì `Rr` và hình dạng giản đồ bức xạ chuẩn hóa không đổi. Điện trở suất thay đổi `Rdc`, `Rac`, từ đó ảnh hưởng đến hiệu suất bức xạ, Gain và phần thực của `Zin`. Khối lượng dây là chỉ số phụ trợ, không tác động đến phép tính RF hiện tại.

Các giá trị điện trở suất thực tế thay đổi theo độ tinh khiết, hợp kim, nhiệt độ và quy trình chế tạo; chương trình chưa mô hình hóa oxide nhôm, lớp mạ, độ nhám bề mặt, mối nối và hiệu ứng lân cận.

## Về AI và cách báo cáo trung thực

Dataset được sinh từ `antenna_core.calculate_antenna`, tức là từ chính mô hình giải tích gần đúng. Random Forest và ANN được huấn luyện để xấp xỉ các nhãn tổng hợp đó. MAE/RMSE/R² vì vậy là độ chính xác của mô hình ML so với **nhãn giải tích**, không phải độ chính xác so với một anten được đo thật hoặc được giải bằng full-wave solver.

Để phát triển thành nghiên cứu có đối chứng mạnh hơn, cần bổ sung dữ liệu đo VNA/anten chamber hoặc dữ liệu mô phỏng full-wave từ CST, HFSS, FEKO, openEMS; sau đó giữ riêng một tập dữ liệu thực nghiệm để đánh giá cuối cùng.

## Các giả thiết và giới hạn của lõi tính toán

- Dùng công thức gần đúng điện trở bức xạ của anten loop nhỏ.
- Dùng điện cảm gần đúng `L₁ ≈ μ₀R[ln(8R/a) − 2]`, nhiều vòng lấy `L ≈ N²L₁`.
- Ước tính tổn hao dây bằng điện trở DC và hiệu chỉnh skin-effect đơn giản.
- `Zin ≈ (Rr + Rac) + jωL`; chưa xét điện dung khe cấp nguồn và tự cộng hưởng.
- Directivity dùng giá trị lý tưởng xấp xỉ 1.5; Gain được suy ra từ hiệu suất.
- Pattern dựa trên hàm Bessel `J₁` cho phân bố dòng đều; hình 3D anten là trực quan, không phải mô hình full-wave.

Không sử dụng các kết quả này như bằng chứng đo lường/chứng nhận thiết kế trước khi kiểm chứng bằng công cụ EM hoặc phép đo thực tế.
