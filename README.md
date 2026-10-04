# Python Project – Sàng lọc hồ sơ tuyển dụng: so sánh KNN, Naive Bayes, SVM và khai phá luật kết hợp (Python)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/<repo>/blob/main/Python_Portfolio/Python_Portfolio/resume_screening_data_mining.ipynb)

> Dự án nhóm 5 người – môn Khai phá dữ liệu, Trường Đại học Ngân hàng TP.HCM.
> **Phần tôi phụ trách:** EDA (phân bố Role/Decision, WordCloud, phân tích độ dài CV)

## Bài toán
1. Phân loại CV vào đúng vị trí công việc (35 vị trí) từ nội dung văn bản.
2. Khai phá luật kết hợp giữa các kỹ năng (Apriori).
3. Tìm hiểu các lý do thường gặp khiến ứng viên bị loại.

## Dữ liệu
- [AzharAli05/Resume-Screening-Dataset](https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset) (Hugging Face): 10.174 CV × 5 cột (Role, Resume, Decision, Reason_for_decision, Job_Description)
- Không có giá trị thiếu hay bản ghi trùng; Role chuẩn hóa từ 45 xuống 35 nhóm
- Dữ liệu không nằm trong repo: tải về và đặt tên `dataset.csv` cùng thư mục với notebook
- Lưu ý: các CV có vẻ do mô hình ngôn ngữ sinh ra, nên kết quả không suy rộng cho CV thật

## Quy trình
1. Làm sạch và chuẩn hóa (lowercase, gộp Role trùng nghĩa, xóa email/số điện thoại/URL, stopwords, loại tên vị trí khỏi CV để tránh rò rỉ nhãn)
2. EDA: phân bố Role, tỷ lệ Select/Reject, WordCloud, độ dài CV theo Role
3. TF-IDF (fit trên tập train) → KNN, Multinomial Naive Bayes, LinearSVC (GridSearchCV, cv=5)
4. Đánh giá bằng accuracy, precision, recall, F1, confusion matrix và phân tích các mẫu bị phân loại sai
5. Apriori trên 65 kỹ năng trích từ CV (7 kỹ năng không nhận diện được, xem Hạn chế)

## Kết quả chính
Chia train/test 80/20 có phân tầng (random_state = 42), test = 2.035 CV:

| Mô hình | Accuracy | Precision | Recall | F1 | Số lỗi | Thời gian |
|---|---|---|---|---|---|---|
| KNN (k = 15) | 98,67% | 0,9861 | 0,9867 | 0,9862 | 27 | ~1,6s |
| Multinomial NB | 96,90% | 0,9656 | 0,9690 | 0,9646 | 63 | ~0,02s |
| LinearSVC (C = 10) | **99,71%** | 0,9972 | 0,9971 | 0,9970 | 6 | ~72s (gồm GridSearchCV) |

- Các vị trí dễ nhầm: Software Developer ↔ Software Engineer (NB), Game Developer ↔ AR/VR Developer (4/6 lỗi của SVM), Data Engineer → Data Architect (KNN), Data Analyst ↔ Data Scientist.
- Apriori: 690 tập phổ biến, 1.250 luật. Nổi bật: {pandas, tensorflow} → {numpy} (confidence 0,94, lift 12,7); {big data, spark} → {hadoop} (confidence 0,97, lift 9,3).

## EDA (phần của tôi)
- Tỷ lệ Select/Reject gần cân bằng (Reject 50,3%).
- Độ dài CV (ký tự, sau khi làm sạch) cao nhất ở AI Researcher/Engineer (median 1.915,5) và thấp nhất ở Game Developer (646, biến động rất lớn).
- 
## Hạn chế
- Dữ liệu có vẻ được sinh tự động (CV mở đầu bằng cùng một câu mẫu), nên accuracy gần 100% khó lặp lại trên CV thật.
- Chỉ một lần chia train/test; k của KNN được chọn theo accuracy trên tập test nên có thể lạc quan.
- Thời gian SVM gồm GridSearchCV (20 lần fit); SVM dùng TF-IDF bigram 5.000 đặc trưng còn KNN/NB dùng từ đơn 2.000 đặc trưng, nên so sánh chưa hoàn toàn công bằng.
- Tên ứng viên và một số mảnh email còn sót trong văn bản sau khi làm sạch.
- Bước kiểm tra mâu thuẫn Decision – Reason chưa được áp dụng trong notebook.
- Apriori: từ điển 65 kỹ năng, nhưng bước làm sạch bỏ ký tự đặc biệt và từ ≤ 2 ký tự nên C++, C#, R, Go, Node.js, scikit-learn, Power BI không nhận diện được; luật kết hợp chỉ phản ánh đồng xuất hiện.
- Tôi chỉ chỉnh sửa phần EDA; các phần còn lại giữ nguyên như bản nhóm đã nộp.

## Cách chạy
1. `pip install -r requirements.txt`
2. Tải `dataset.csv` từ link ở mục Dữ liệu, đặt cùng thư mục với notebook.
3. Chạy notebook từ trên xuống (hoặc bấm nút Colab).
