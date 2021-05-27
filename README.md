# Improving Abstractive Summarization with Curriculum Learning
  + Authors: Jinhyeong Lim, Hyeon-Je Song
  + Paper(To appear in KCC2021)

<br>
생성요약의 성능을 향상 시키기 위해 상대적으로 쉬운 과제인 추출요약 학습 후 어려운 과제인 생성요약을 학습 시키는 커리큘럼 학습 사용 
  
<br>
추출 요약문(Reference Extractive-Summary)가 없는 경우를 대비해 임시 추출요약문 (Pseudo Extractive-Summary) 를 만들기 위해 TextRank, Lead-N, Principal 전략 을 사용하여 커리큘럼 학습 진행
<br>

<br> ![캡처](https://user-images.githubusercontent.com/64317686/119776563-0fbecd80-bf00-11eb-9225-56c05ec67844.JPG)
-  총 학습 횟수를 __K번__ 으로 고정
-  사전학습 모델에 본문을 입력으로 __추출 요약문__ 을 생성하는 과제를 __A번__ 학습
-  1차 미세조정 된 모델에 본문을 입력으로 __추상 요약문__ 을 생성하도록 __B (K - A)번__ 학습
<br><br>

### Requirements
- Python 3.7+
  - __pip install -r requirements.txt__ or manually install the packages below.
```
torch==1.8.1
transformers==4.6.0
json
rouge
pandas
numpy
summa
torch.utils.data
```
### Data
- 국립 국어원 요약 말뭉치 데이터셋, 신문 말뭉치 데이터셋 사용
- 본문은 신문 말뭉치 데이터셋, 요약문은 요약 말뭉치 데이터셋 에서 Parsing 
- 총 데이터 4,387개 8:1:1 비율로 나누어 3,509개의 학습 데이터, 439개의 검증 데이터, 439개의 평가 데이터를 사용해 학습 진행



