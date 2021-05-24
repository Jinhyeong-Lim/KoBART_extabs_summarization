from rouge import Rouge
import torch


def eval(model, tokenizer, test_loader, device):
    """
    Model Evaluation
    """
    r = Rouge()
    total_r1 = 0
    total_rl = 0
    total_r2 = 0
    test_len = len(test_loader)

    model.to(device)
    model.eval()

    with torch.no_grad():
        k = 0
        for text, ans in test_loader:

            # encoding and zero padding
            test_doc = [tokenizer.encode_plus(t, add_special_tokens=True,
                                              max_length=1024,
                                              pad_to_max_length=True)
                        ["input_ids"] for t in text]

            # tensor, gpu
            test_doc = torch.tensor(test_doc)
            test_doc = test_doc.to(device)

            # Generate Summarizaion
            summary_ids = model.generate(test_doc,
                                         num_beams=5,
                                         no_repeat_ngram_size=4,
                                         temperature=1.0, top_k=-1, top_p=-1,
                                         length_penalty=1.0, min_length=1,
                                         max_length=100
                                         ).to(device)

            # Summarization Preprocessing
            output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            output = output[:len(output) - output[::-1].find('.')]

            # Print
            print("Original_Article : ", str(text[0]))
            print("\nReference_Summary : ", str(ans[0]))
            print("\nModel_Summary: \n", output)

            # Calculate Rouge Score
            ref = tokenizer.tokenize(str(ans[0]))
            virsum = tokenizer.tokenize(str(output))

            scores = r.get_scores(str(virsum)[1:-1], str(ref)[1:-1])
            rouge1, rouge2, rougel = scores[0]['rouge-1']['f'], scores[0][
                'rouge-2']['f'], scores[0]['rouge-l']['f']

            total_r1 += float(rouge1)
            total_r2 += float(rouge2)
            total_rl += float(rougel)

            print("\nRouge-1 : " + str(rouge1) + "\nRouge-2 : " + str(rouge2) +
                  "\nRouge-L : " + str(rougel))

    return total_r1 / test_len, total_r2 / test_len, total_rl / test_len
