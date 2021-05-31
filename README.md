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
```
### Data
- 국립 국어원 요약 말뭉치 데이터셋, 신문 말뭉치 데이터셋 사용
- 본문은 신문 말뭉치 데이터셋, 요약문은 요약 말뭉치 데이터셋 에서 Parsing 
- 총 데이터 4,387개 8:1:1 비율로 나누어 3,509개의 학습 데이터, 439개의 검증 데이터, 439개의 평가 데이터를 사용해 학습 진행

### Run
- ##### Train 
  -  사전학습 모델 KoBART를 이용해 Fine-tuning
  -  arguments
      -  strategy (Choose strategy ['textrank', 'principal', 'lead_n'], default=None(Reference extractive summary))
      -  ext_epcohs
      -  abs_epochs
      -  seed (Random seed)
        ```
        python main.py --strategy "?" --exp_epochs "?" --abs_epcosh "?" --seed "?"
        ```
 - ##### Evaluation
    - Model이 생성한 Abstractive Summray와 Reference Abstractive Summray 사이 Rouge1,2,L Score 계산 
    - Rouge Score를 계산하기 위해 kobart_tokenizer 사용

    ![캡처](https://user-images.githubusercontent.com/64317686/120109519-b925ed80-c1a4-11eb-9dc6-bd451f0df4cb.JPG)
    
 - ##### Conclusion
    - 추출요약을 먼저 학습 시킨 후 생성요약을 학습하는 __커리큘럼 학습__ 방법이 __생성요약 성능 향상__ 에 도움을 준다.
    - 추출 요약문이 없는 경우를 대비하여 임시 추출 요약문을 만들기 위해 __TextRank, Lead-N, Pincipal 전략 사용__ 했다.
    - __Lead_N 전략을 사용해 커리큘럼 학습을 진행한 모델__ 과 정답 추출 요약문을 사용해 커리큘럼 학습을 진행한 모델을 비교했을 때 __준수한 성능__ 을 보인다.

