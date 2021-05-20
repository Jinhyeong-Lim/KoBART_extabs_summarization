import json

ext_data = []
abs_data = []

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
            doc_id = str(d["document_id"]).strip().split(".")
            ext=""
            abs=""

            # 정답 추출요약문 데이터 Parsing
            for i in range(len(d["topic_sentences"])):
                ext = ext + d["topic_sentences"][i] + " "

            # 정답 추상요약문 데이터 Parsing
            for i in range(len(d["summary_sentences"])):
                abs = abs + d["summary_sentences"][i] + " "

            # 해당 기사 본문 Parsing
            with open('./news_data/data/'+ str(doc_id[0])+".json") as json_file:
                            origin_data = json.load(json_file)
                            origin_data = origin_data["document"]
                            for data in origin_data:
                                if data["id"] == d["document_id"]:
                                    paragraph = data["paragraph"]
                                    preprocess_text = ""
                                    for sentence in paragraph:
                                        preprocess_text = preprocess_text  + " "+ str(sentence["form"])
                                    ext_data.append((preprocess_text,ext))
                                    abs_data.append((preprocess_text,abs))
                                    break

    return abs_data

