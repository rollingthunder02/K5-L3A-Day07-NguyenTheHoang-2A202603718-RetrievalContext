# Báo cáo Lab Day 7 — Nguyễn Thế Hoàng (2A202603718)

## Dữ liệu và cách đo dev

Dùng quy chế HUST 2025, snapshot B của bảng `day06_chunks` với 445 chunk. Dense dùng pgvector cosine và `text-embedding-3-small` 1536 chiều. Agent `gpt-4o-mini` dùng tool `search`. Mọi cấu hình dùng cùng 21 câu dev (19 có đáp án và 2 thiếu bằng chứng), top-k 3 mỗi lượt và không có filter. Hybrid lấy 10 ứng viên mỗi nhánh và dùng RRF với hằng số 60.

Hit@1 và Hit@3 tính riêng trên ranking của câu hỏi gốc. Cột Tool Hit@3 tính evidence xuất hiện trong top 3 của ít nhất một lượt gọi tool. Context Evidence Rate kiểm tra evidence trong context cuối. Mẫu số của ba phép đo retrieval và phép đo context là 19 câu có đáp án. Câu trả lời và citation được đọc đối chiếu với đáp án và đoạn nguồn.

## Kết quả dev

| Cấu hình | Budget / lượt search tối đa | Hit@1 | Hit@3 | Tool Hit@3 | Context Evidence Rate | Trung vị tổng (ms) | Chat ước tính (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense-base | 800 / 1 | 12/19 (63,2%) | 16/19 (84,2%) | 10/19 (52,6%) | 9/19 (47,4%) | 3.044 | 0,003211 |
| tfidf-base | 800 / 1 | 0/19 (0%) | 1/19 (5,3%) | 3/19 (15,8%) | 3/19 (15,8%) | 1.957 | 0,002646 |
| hybrid-base | 800 / 1 | 9/19 (47,4%) | 14/19 (73,7%) | 10/19 (52,6%) | 9/19 (47,4%) | 3.221 | 0,003251 |
| hybrid-budget | 2.000 / 1 | 9/19 (47,4%) | 14/19 (73,7%) | 10/19 (52,6%) | 10/19 (52,6%) | 3.149 | 0,003424 |
| hybrid-multisearch | 2.000 / 3 | 9/19 (47,4%) | 14/19 (73,7%) | 7/19 (36,8%) | 7/19 (36,8%) | 5.832 | 0,006576 |

| Cấu hình | Điểm 0 / 1 / 2 | Trung bình /2 | Citation hỗ trợ | Từ chối đúng | Từ chối nhầm |
|---|---:|---:|---:|---:|---:|
| dense-base | 10 / 1 / 8 | 0,895 | 8/9 | 2/2 | 10/19 |
| tfidf-base | 16 / 0 / 3 | 0,316 | 3/4 | 2/2 | 16/19 |
| hybrid-base | 10 / 0 / 9 | 0,947 | 9/9 | 2/2 | 10/19 |
| hybrid-budget | 9 / 0 / 10 | 1,053 | 10/12 | 2/2 | 7/19 |
| hybrid-multisearch | 12 / 2 / 5 | 0,632 | 7/9 | 2/2 | 11/19 |

Cả năm cấu hình cuối gọi tool ở 21/21 câu và không có lỗi chạy. Bốn cấu hình một lượt gọi đúng một lần mỗi câu. Cấu hình nhiều lượt gọi 1, 2 và 3 lần trên lần lượt 7, 1 và 13 câu. Chi phí chat tính từ token input và output theo bảng giá tham chiếu ngày 14/11/2026, tương ứng 0,15 và 0,60 USD mỗi triệu token. Runner chưa lưu usage embedding, nên bảng chưa gồm khoản đó. Các cấu hình chạy song song, vì vậy độ trễ phản ánh điều kiện chạy này. [Bảng giá GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini)

## Hai thử nghiệm cải thiện và phân tích lỗi

1. **Tăng budget từ 800 lên 2.000 ký tự.** Giả thuyết là giữ được thêm đoạn khi ba hit cùng dài. So với hybrid-base, context đủ evidence tăng từ 9/19 lên 10/19. Ở q02, agent tìm bằng truy vấn tiếng Anh và đưa đoạn `7:1693` vào context, từ đó trả lời đúng 3–4 năm. Hai lần chạy có query khác nhau, nên không thể quy toàn bộ mức tăng cho budget.
2. **Tăng giới hạn từ 1 lên 3 lượt search.** Giả thuyết là truy vấn bổ sung tìm lại evidence còn thiếu. Kết quả context giảm từ 10/19 xuống 7/19, điểm grounded giảm từ 1,053 xuống 0,632 và thời gian tăng. Ở q20 cấu hình nhiều lượt lấy được đoạn `24:1428` và trả lời đúng, trong khi hybrid-budget trích `24:992`, một tiêu đề không chứa điều kiện hoàn thành học phần. Ở q19, lượt tìm của cấu hình nhiều lượt đưa điều khoản về lộ trình học trước vào context và agent trả lời nhầm 15 tín chỉ. Nguồn đúng là `24:87`, nêu 24 tín chỉ mỗi học kỳ chính và 45 tín chỉ mỗi năm.
3. **q10 — evidence rớt khi đóng gói và nhầm hệ đào tạo.** Dense lấy đúng đoạn `13:1596` ở hạng 3, nhưng block rớt khỏi context 800 ký tự và agent từ chối. Hybrid-budget tra được một đoạn có mức 12 tín chỉ ở `21:90` nhưng đoạn này thuộc hệ kỹ sư. Cấu hình nhiều lượt giữ cả đoạn đúng và đoạn sai hệ, dẫn đến câu trả lời chỉ được điểm 1. TF-IDF tra bằng tiếng Anh và dùng đúng nguồn trang 13.
4. **Lỗi định dạng ở lần chạy nhiều lượt đầu tiên.** 7/21 câu lỗi parse JSON. Controller đã được sửa để yêu cầu JSON ở mọi lượt có thể trả lời cuối. Lần chạy lại xử lý đủ 21 câu và không có lỗi parse. Các cấu hình một lượt dùng cùng yêu cầu JSON ở lượt cuối trước và sau sửa. Trace lần đầu lưu ở `starter/results/dev-hybrid-multisearch/initial-traces.json`.

## Cấu hình đề xuất cho phần 6

Chọn theo điểm grounded trung bình, sau đó đến độ phủ evidence trong context và độ trễ/chi phí khi chất lượng tương đương. Theo dữ liệu dev, **hybrid-budget** đứng đầu về grounded answer (1,053/2) và context (10/19), dù dense-base có ranking query gốc tốt nhất. Chạy test ở phần 6 với `dense-base` và `hybrid-budget` theo cấu hình đã đo. Bộ test Day 7 chưa được chạy trong phần này.

Mỗi thư mục `starter/results/dev-<config>/` có `traces.json`, `reviews.json` và `summary.json`. Từ thư mục Day 7, đặt `$day6 = (Get-Item ..).FullName` rồi chạy `python -X utf8 starter/benchmark.py --day6-root $day6 --split dev --config <config> --output starter/results/dev-<config>`. Dùng các ID cấu hình trong bảng.

## Phần 6 — Kết quả test đã giữ riêng

Chốt từ dev và chạy trên cùng 7 câu test. Baseline dùng dense, budget 800 ký tự. Cấu hình chọn dùng hybrid RRF, budget 2.000 ký tự. Cả hai dùng snapshot B, `gpt-4o-mini`, top-k 3, tối đa một lượt search và không áp dụng filter. Các cấu hình giữ nguyên sau khi có kết quả test.

| Cấu hình | Hit@1 câu gốc | Hit@3 câu gốc | Context đủ evidence | Context 0 / 1 / 2 | Điểm trả lời 0 / 1 / 2 | Trung bình /2 |
|---|---:|---:|---:|---:|---:|---:|
| dense-base | 2/5 (40%) | 3/5 (60%) | 1/5 (20%) | 4 / 0 / 1 | 4 / 0 / 1 | 0,40 |
| hybrid-budget | 0/5 (0%) | 2/5 (40%) | 1/5 (20%) | 3 / 1 / 1 | 4 / 0 / 1 | 0,40 |

| Cấu hình | Tool use | Citation hợp lệ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm | Lỗi | Trung vị tổng (ms) | Chat ước tính (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense-base | 7/7 | 1/2 | 2/2 | 2/2 | 3/5 | 0 | 3.643 | 0,001052 |
| hybrid-budget | 7/7 | 1/1 | 1/1 | 2/2 | 4/5 | 0 | 3.711 | 0,001184 |

Mỗi cấu hình gọi search đúng một lần trên mỗi câu. Chi phí chat tính theo cách ở phần dev, chưa gồm embedding. Tỷ lệ citation hợp lệ tính trên số citation phát ra.

| ID | dense-base | hybrid-budget |
|---|---|---|
| T01 | Đúng, trích trang 10 về giữ điểm cao nhất khi học lại. | Từ chối, context chỉ có các đoạn về CPA để tốt nghiệp. |
| T02 | Từ chối, context thiếu hạn thi bù và điểm F. | Từ chối, context thiếu cùng điều khoản. |
| T03 | Từ chối. Ranking câu gốc chạm evidence trong top 3, nhưng lượt search do agent chọn không tìm được đoạn đó. | Từ chối, cũng có evidence trong ranking gốc và thiếu trong lượt search của agent. |
| T06 | Từ chối sau khi lượt search của agent tìm các đoạn về luận văn thạc sĩ; query gốc có evidence ở hạng 1. | Đúng, trích đoạn `30:2062` hỗ trợ ngưỡng từ 5,5/10. |
| T07 | Sai, trả lời hai thành viên tán thành dựa vào điều khoản về phiếu phản đối ở cấp trường. Điều khoản cấp cơ sở ở trang 32 yêu cầu ít nhất 6/7 phiếu thuận. | Từ chối. Context có thông tin hội đồng cấp cơ sở gồm 7 người nhưng thiếu ngưỡng số phiếu thuận. |
| TU01 | Tra cứu và từ chối đúng vì thiếu mức phí phúc khảo. | Tra cứu và từ chối đúng. |
| TU02 | Tra cứu và từ chối đúng vì thiếu yêu cầu về dịch thuật và công chứng. | Tra cứu và từ chối đúng. |

Ưu thế về điểm câu trả lời của hybrid-budget trên dev không lặp lại trên test. Hai cấu hình trả lời đúng các câu khác nhau. Dense xếp hạng câu gốc tốt hơn, trong khi việc đổi query của agent có thể làm mất evidence như ở T03 và T06. Bộ test chỉ có 5 câu có đáp án, mỗi câu tương đương 20 điểm phần trăm trong chỉ số retrieval và context.

[Tệp kết quả dense-base](../../starter/results/test-dense-base/summary.json), [trace dense-base](../../starter/results/test-dense-base/traces.json), [nhãn dense-base](../../starter/results/test-dense-base/reviews.json).  
[Tệp kết quả hybrid-budget](../../starter/results/test-hybrid-budget/summary.json), [trace hybrid-budget](../../starter/results/test-hybrid-budget/traces.json), [nhãn hybrid-budget](../../starter/results/test-hybrid-budget/reviews.json).
