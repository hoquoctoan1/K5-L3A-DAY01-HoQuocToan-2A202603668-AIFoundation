# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác

- **SEED:** 5953.
- **Nguồn số liệu:** output trong `day01_lab.ipynb` và `lab1_report.json` đi kèm.
- **Phạm vi:** đã có kết quả ML, DL, embedding và thử thách #2; chưa có kết quả thực nghiệm LLM do API hết số dư.

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| [day01_lab.ipynb](day01_lab.ipynb) | Có mã, output, biểu đồ và kết quả kiểm tra |
| [lab1_report.json](lab1_report.json) | 15/15 kiểm tra đạt; đủ 10 dự đoán; SEED 5953; thử thách #2 |
| [hand_calc.jpg](hand_calc.jpg) | Ảnh viết tay có các bước của đủ 5 phép tính |

Dữ liệu Kaggle ban đầu có 5.572 tin. Sau bỏ trùng theo văn bản còn 5.169 tin, gồm 4.516 ham và 653 spam. Tập train có 3.619 tin, validation 775 tin, test 775 tin; test có 99 spam. SUB gồm 40 tin cân bằng để so sánh các phương án.

Kết quả tính tay: `baseline_accuracy = 0.87`, `sigmoid_0 = 0.50`, `f1_example = 0.73`, `cosine_example = 0.60`, `llm_cost_per_day = 10.50` USD/ngày theo giá giả định trong đề. Cả 5 giá trị đều qua kiểm tra.

Các kiểm tra mã LLM sử dụng API giả lập. Hai ô thí nghiệm LLM thật đã trả `429`, `insufficient_quota`, `credit_balance_exhausted`. Vì vậy, 15/15 kiểm tra đạt không có nghĩa là đã đo thành công tất cả phương án LLM; những hàng chưa đo được ghi rõ ở dưới.

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1.000 tin (API) |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.9312 | 0.8889 | 0.1001 | 0 |
| BoW + MLP | 0.9418 | 0.8889 | 0.3183 | 0 |
| MiniLM + kNN | 0.9275 | 0.9500 | 22.0855 | 0 |
| MiniLM + LogReg | 0.9538 | 0.9189 | 12.1408 | 0 |
| MiniLM + MLP head | 0.9490 | 0.9189 | 42.0121 | 0 |
| LLM nhỏ zero-shot | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| LLM lớn zero-shot | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| LLM nhỏ few-shot | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

| Phương án | Train (giây) | Thông lượng theo lô (tin/giây) |
| --- | ---: | ---: |
| BoW + LogReg | 3.2243 | 47.263,31 |
| BoW + MLP | 4.4809 | 31.693,86 |
| MiniLM + kNN | 2.6083 | 104,75 |
| MiniLM + LogReg | 3.1419 | 1.160,65 |
| MiniLM + MLP head | 5.4586 | 522,71 |

MiniLM + LogReg có F1 cao nhất trên toàn bộ test trong lần chạy này. MiniLM + kNN có F1 cao nhất trên SUB 40. Hai tập có phân phối và kích thước khác nhau nên không dùng hai cột này thay thế nhau. Chênh lệch nhỏ trên SUB chưa đủ để kết luận mô hình nào tốt hơn nói chung. Giá trị API bằng 0 của các mô hình cục bộ chưa bao gồm chi phí máy chạy.

## 3. Dự đoán và thực tế

Ba dự đoán có chênh lệch hoặc không được xác nhận rõ trong kết quả:

| Dự đoán đã ghi | Thực tế | Phân tích |
| --- | --- | --- |
| `ml_f1`: F1 khoảng 0.82; precision khoảng 0.90 và recall khoảng 0.75 | F1 0.9312; precision 0.9778; recall 0.8889 | F1 cao hơn dự đoán khoảng 0.1112. BoW khai thác được nhiều tín hiệu từ vựng trong bộ dữ liệu này. Dự đoán precision cao hơn recall vẫn đúng. |
| `ml_lr`: learning rate 50 làm loss tăng giảm bất thường, khó hội tụ | Biểu đồ có đỉnh tăng mạnh lúc đầu, sau đó loss giảm và hội tụ về khoảng 0.006; learning rate 0.001 còn loss khoảng 0.627 | Dự đoán đúng về cú cập nhật quá mạnh ban đầu nhưng không đúng nếu kết luận mất ổn định kéo dài. Trong lần chạy này, learning rate 50 vẫn hội tụ. Không suy rộng rằng learning rate lớn luôn tốt. |
| `emb_tricky`: Encoder + LogReg đoán đúng nhiều hơn BoW | Cả hai đúng 5/6 tin | Cả hai đều nhầm tin nhắc phần thưởng tài khoản sắp hết hạn thành ham. Biểu diễn theo nghĩa không tự bảo đảm nhận ra ý đồ lừa đảo. |

Các dự đoán `llm_models` và `llm_fewshot` đã được ghi nhưng chưa thể đối chiếu với kết quả thực nghiệm vì API lỗi hết số dư. Không coi các dự đoán này là kết quả đo.

## 4. Phân tích 50 case khó

| Mô hình | F1 SUB thường | F1 trên 50 case khó | Mức giảm F1 | Accuracy trên 50 case |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.889 | 0.500 | 0.389 | 56% |
| BoW + MLP | 0.889 | 0.524 | 0.365 | 60% |
| MiniLM + kNN | 0.950 | 0.737 | 0.213 | 70% |
| MiniLM + LogReg | 0.919 | 0.612 | 0.307 | 62% |
| MiniLM + MLP head | 0.919 | 0.625 | 0.294 | 64% |

BoW + LogReg giảm F1 nhiều nhất khi đối chiếu với SUB thường. Cả hai phương án BoW chỉ đúng khoảng 17% ở nhóm né từ khóa và 20% ở nhóm spam giọng tự nhiên. Tách chữ hoặc thay ký tự làm mất các token quen thuộc; lời nhắn tự nhiên cũng ít chứa từ khóa quảng cáo mạnh.

MiniLM + kNN đúng khoảng 83% nhóm né từ khóa và 100% nhóm thao túng/lừa đảo, nhưng chỉ đúng 25% nhóm nói về spam. MiniLM + LogReg chỉ đúng 20% nhóm spam giọng tự nhiên và khoảng 38% nhóm thao túng. Như vậy, cùng một encoder vẫn có thể cho kiểu lỗi khác nhau tùy đầu phân loại.

Notebook còn lưu output của 5 case bổ sung, đánh giá riêng sau bộ 50 case. Hai ví dụ cụ thể có thể đối chiếu trực tiếp với output này là:

1. **Spam tách chữ:** `Fr e e ca sh re ward, re ply n ow to unlock your gift`. Cả 5 mô hình đều dự đoán ham. Cách tách chữ làm yếu tín hiệu từ vựng và cũng khiến encoder không khôi phục được ý nghĩa đủ tốt trong trường hợp này.
2. **Ham trích dẫn spam trong lớp học:** `Our security class analysed 'Claim your FREE cash prize now'. Do not follow it.` Cả 5 mô hình đều dự đoán spam. Các từ quảng cáo trong đoạn trích lấn át ngữ cảnh đang phân tích và cảnh báo, gây chặn nhầm.

Hai ví dụ trên thuộc tập bổ sung, không cộng vào chỉ số của 50 case gốc. Các case bổ sung được notebook ghi nhận là do AI đề xuất.

Điểm trên test thường có thể lạc quan vì dữ liệu còn các mẫu gần giống nhau dù đã bỏ trùng tuyệt đối. Bộ case khó giúp phát hiện điểm yếu; đây không phải mẫu đại diện để ước lượng tỷ lệ lỗi ngoài đời. Chưa có phép đo LLM nên chưa kết luận được hiệu quả của prompt phòng thủ.

## 5. Thử thách cá nhân

**Đề #2, SEED 5953:** giảm vocab xuống 100 từ và tăng lên 5.000 từ; train lại, báo F1 và giải thích xu hướng.

Thí nghiệm đã lưu giữ nguyên split, learning rate 50 và 300 epoch. Vocab được xây từ tập train; sau đó vector hóa và huấn luyện lại LogReg.

| Số từ | Accuracy test | Precision | Recall | F1 test | Thời gian trong ô thử thách (giây) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.9729 | 0.9333 | 0.8485 | 0.8889 | 0.3512 |
| 2.000 — mô hình gốc | 0.9832 | 0.9778 | 0.8889 | 0.9312 | 3.2243 |
| 5.000 | 0.9832 | 0.9778 | 0.8889 | 0.9312 | 11.2100 |

Vocab 100 bỏ bớt các từ hữu ích nên recall và F1 thấp hơn. Tăng lên 5.000 từ không tăng F1 so với 2.000 từ trong lần đo này, nhưng tăng chi phí tính toán. Với các kết quả đã quan sát, 2.000 từ là lựa chọn cân bằng hơn. Thời gian ô thử thách bao gồm bước dự đoán sau train nên chỉ so sánh gần đúng với thời gian train riêng của mô hình gốc.

**Nguồn hỗ trợ:** notebook ghi rõ mã và phần giải thích thử thách đã có AI hỗ trợ. Báo cáo giữ nguyên thông tin này; không trình bày đây là kết quả làm độc lập không dùng AI.

## 6. Phương án cho sản phẩm thật

Tải 10 triệu SMS/ngày tương đương khoảng 115,74 tin/giây trung bình. Cả 5 mô hình cục bộ đều có p95 dưới 50 ms/tin trong benchmark đã lưu.

**Đề xuất ưu tiên MiniLM + LogReg để thử nghiệm tiếp:** F1 test 0.9538, p95 12.1408 ms/tin và thông lượng theo lô khoảng 1.160,65 tin/giây. Phương án này có F1 test cao nhất trong bảng và còn khoảng cách so với giới hạn độ trễ. Tuy nhiên, F1 trên case khó chỉ 0.612; cần cải thiện và đánh giá trên dữ liệu thực tế trước triển khai.

BoW + LogReg có p95 khoảng 0.10 ms/tin và thông lượng cao, phù hợp nếu ưu tiên chi phí tính toán thấp. MiniLM + kNN có F1 SUB và case khó tốt hơn các phương án khác trong lần chạy này, nhưng thông lượng khoảng 104,75 tin/giây thấp hơn tải trung bình; cấu hình hiện tại cần tăng năng lực xử lý trước khi đáp ứng yêu cầu sản lượng. Thông lượng theo lô cũng không thay thế phép thử tải có hàng đợi và lưu lượng tăng đột biến.

Một hướng kết hợp là để mô hình cục bộ xử lý các tin chắc chắn, chuyển các tin có xác suất ở vùng lưỡng lự sang bước kiểm tra bổ sung hoặc LLM. Vùng [0.3, 0.7] chỉ là điểm khởi đầu, phải chọn ngưỡng trên validation và cân nhắc chặn nhầm OTP. Chưa có số đo LLM thật nên chưa xác nhận được độ trễ hay chất lượng của hệ thống kết hợp.

Theo giả định giá và token trong đề: model nhỏ tốn 10.50 USD cho 1 triệu SMS, nên 10 triệu SMS/ngày tốn 105 USD/ngày, tương đương **3.150 USD/30 ngày** nếu dùng cho toàn bộ tin. Đây là phép tính lý thuyết, chưa tính prompt thực tế, retry, hạ tầng hoặc tỷ lệ tin được chuyển sang LLM.

## 7. AI log và điều đã học

| Phần / yêu cầu | Hỗ trợ đã được ghi nhận | Kiểm chứng và giới hạn |
| --- | --- | --- |
| TODO 3.2–3.3 | AI log trong notebook ghi AI hỗ trợ top-k và kNN | Có output kiểm tra, benchmark và thí nghiệm 6 tin |
| TODO 4.1–4.4 | AI hỗ trợ mã gọi API, prompt, parser và batch | Kiểm tra giả lập đạt; gọi API thật lỗi hết số dư |
| Phần 5 | AI hỗ trợ dự đoán, phân tích, 5 case bổ sung và thử thách vocab | Có output số liệu; các phần có AI hỗ trợ đã được ghi rõ trong notebook |
| Tính tay | Hỏi công thức, kết quả và yêu cầu giấy/Word | Đã chép lên giấy, gửi ảnh; 5 kết quả qua kiểm tra |
| Nộp bài và báo cáo | Hỏi cách tạo repo, upload và hoàn thành báo cáo từ ba tệp | Báo cáo sử dụng số liệu trong tệp; không bổ sung số đo LLM chưa có |
| Rà soát lần cuối | AI bổ sung giải thích data leakage, đối chiếu biểu đồ learning rate và chỉnh giải thích MLP/overfitting | Chỉ sửa phần chữ của bản notebook nộp; không chạy lại thí nghiệm hoặc thay output |

Qua các kết quả, accuracy cao chưa đủ để đánh giá bộ lọc spam: mô hình luôn đoán ham đã đạt khoảng 87.37% nhưng bỏ sót toàn bộ spam. Embedding giúp cải thiện nhiều trường hợp, song vẫn mắc lỗi ngữ cảnh và lừa đảo; cần xem từng nhóm lỗi bên cạnh F1 tổng thể. Kiểm tra giả lập xác nhận một phần hoạt động của mã, còn chất lượng và chi phí thực tế của LLM phải được đo qua API thật.

**Ghi chú về việc hoàn thiện báo cáo:** nội dung này được AI tổng hợp từ các tệp và hội thoại đã cung cấp; người nộp cần đọc, kiểm chứng và có khả năng giải thích các kết luận.
