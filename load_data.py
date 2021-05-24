import json


def data_load():
    """
    Loads data
    """
    ext_data = []
    abs_data = []
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
            with open('./news_data/data/' + str(doc_id[0])+".json") as \
                    j_file:
                origin_data = json.load(j_file)
                for data in origin_data["document"]:
                    if data["id"] == d["document_id"]:
                        paragraph = data["paragraph"]
                        preprocess_text = " ".join(str(
                                sentence["form"]) for sentence in paragraph)
                        ext_data.append((preprocess_text, ext))
                        abs_data.append((preprocess_text, abs))
                        break

    return abs_data
