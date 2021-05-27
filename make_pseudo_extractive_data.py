import json
from rouge import Rouge
from kobart_transformers import get_kobart_tokenizer
from summa import summarizer

tokenizer = get_kobart_tokenizer()
r = Rouge()


def textrank():
    """
    TextRank 전략을사용해 임시 추출 요약문 생성
    """

    textrank_data = []
    with open('summary_data.json', "r") as json_file:
        summary = json.load(json_file)
        summary = summary["data"]
        for d in summary:
            # 해당 기사 id Parsing
            doc_id = str(d["document_id"]).strip().split(".")

            # 해당 기사 본문 Parsing
            with open('./news/data/' + str(doc_id[0])+".json") as \
                    j_file:
                origin_data = json.load(j_file)
                for data in origin_data["document"]:
                    if data["id"] == d["document_id"]:
                        paragraph = data["paragraph"]
                        preprocess_text = " ".join(str(
                                sentence["form"]) for sentence in paragraph)
                        textrank_summary = summarizer.summarize(
                            preprocess_text, ratio=0.2)
                        textrank_data.append((preprocess_text,
                                              textrank_summary))
                        break

    return textrank_data


def principal():
    """
    Principal 전략을사용해 임시 추출 요약문 생성
    """

    principal_data = []
    with open('summary_data.json', "r") as json_file:
        summary = json.load(json_file)
        summary = summary["data"]
        for d in summary:
            # 해당 기사 id Parsing
            doc_id = str(d["document_id"]).strip().split(".")

            # 해당 기사 본문 Parsing
            with open('./news/data/' + str(doc_id[0])+".json") as \
                    j_file:
                origin_data = json.load(j_file)
                for data in origin_data["document"]:
                    if data["id"] == d["document_id"]:
                        paragraph = data["paragraph"]
                        preprocess_text = " ".join(str(
                                sentence["form"]) for sentence in paragraph)

                        # 문장 구분
                        k = preprocess_text.split('다. ')
                        for sent in range(len(k) - 1):
                            k[sent] = k[sent] + "다."
                            if k[sent][0] == " ":
                                k[sent] = k[sent][1:]

                        # 각 sentence와 remain sentence 간의 rouge1 score 계산
                        m = {}
                        for j in range(len(k)):
                            rest = ""
                            tar = k[j]
                            for w in range(len(k)):
                                if w != j:
                                    rest = rest + k[w]
                            scores = r.get_scores(str(tokenizer.tokenize(tar)),
                                                  str(tokenizer.tokenize(rest)))
                            rouge1 = scores[0]['rouge-1']['f']
                            m[j] = rouge1

                        # rouge score에 따라 정렬
                        n = sorted(m.items(), key=lambda item: item[1],
                                   reverse=True)

                        # rouge 1 socore가 높은 문장 추출 (p=0.2)
                        principal_summary = " ".join(k[n[i][0]] for i in
                                                     range(int(len(k) * 0.2)))

                        principal_data.append((preprocess_text,
                                              principal_summary))
                        break

    return principal_data


def lead_n():
    """
    Lead-N 전략을사용해 임시 추출 요약문 생성
    """

    lead_data = []
    with open('summary_data.json', "r") as json_file:
        summary = json.load(json_file)
        summary = summary["data"]
        for d in summary:
            # 해당 기사 id Parsing
            doc_id = str(d["document_id"]).strip().split(".")

            # 정답 추출요약문 데이터 Parsing
            ext = " ".join(i for i in d["topic_sentences"])

            # 정답 추상요약문 데이터 Parsing
            abs = " ".join(i for i in d["summary_sentences"])

            # 해당 기사 본문 Parsing
            with open('./news/data/' + str(doc_id[0])+".json") as \
                    j_file:
                origin_data = json.load(j_file)
                for data in origin_data["document"]:
                    if data["id"] == d["document_id"]:
                        paragraph = data["paragraph"]
                        preprocess_text = " ".join(str(
                                sentence["form"]) for sentence in paragraph)

                        # 문장 구분
                        k = preprocess_text.split('다. ')
                        for sent in range(len(k) - 1):
                            k[sent] = k[sent] + "다."
                            if k[sent][0] == " ":
                                k[sent] = k[sent][1:]

                        # 본문 서두의 문장 추출
                        lead_summary = " ".join(k[i] for i in
                                                range(int(len(k) * 0.2)))

                        lead_data.append((preprocess_text, lead_summary))
                        break

    return lead_data

