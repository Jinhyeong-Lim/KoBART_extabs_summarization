import torch


def train(train_loader, valid_loader, epochs, model,
          tokenizer, optimizer, device):
    """
    Training
    """
    total_loss = 0
    best_valid_loss = 987654321
    itr = 1
    p_itr = 500
    for epoch in range(epochs):
        for text, ans in train_loader:

            # encoding and zero padding
            ori_doc = [tokenizer.encode_plus(t, add_special_tokens=True,
                                             max_length=1024,
                                             pad_to_max_length=True)[
                           "input_ids"] for t in text]
            ref_sum = [tokenizer.encode_plus(t, add_special_tokens=True,
                                             max_length=512,
                                             pad_to_max_length=True)[
                           "input_ids"] for t in ans]

            # decoder_inputs : <pad> + reference_summary
            dec_in = [3]
            for i in ref_sum[0][:-1]:
                dec_in.append(i)

            # labels : reference_summary + <eos> token
            if ref_sum[0].index(3):
                ref_sum[0][ref_sum[0].index(3)] = 0
            else:
                ref_sum[0][-1] = 1

            # tensor, gpu
            ori_doc = torch.tensor(ori_doc)
            dec_in = torch.tensor([dec_in])
            ref_sum = torch.tensor(ref_sum)
            ori_doc = ori_doc.to(device)
            dec_in = dec_in.to(device)
            ref_sum = ref_sum.to(device)

            # Training, optimization, loss Fuction
            outputs = model(ori_doc, decoder_input_ids=dec_in, labels=ref_sum)
            optimizer.zero_grad()
            total_loss += outputs.loss
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if itr % p_itr == 0:
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}'.format
                      (epoch + 1, epochs, itr, total_loss / p_itr))

                total_loss = 0

            itr += 1

        # Validation data evaluation
        model.eval()
        with torch.no_grad():
            for val_text1, val_ans1 in valid_loader:

                # encoding and zero padding
                valid_doc = [tokenizer.encode_plus(t, add_special_tokens=True,
                                                   max_length=1024,
                                                   pad_to_max_length=True)
                             ["input_ids"] for t in val_text1]
                valid_summary = [tokenizer.encode_plus(t,
                                                       add_special_tokens=True,
                                                       max_length=512,
                                                       pad_to_max_length=True)
                                 ["input_ids"] for t in val_ans1]

                # labels : reference_summary + <eos> token
                if valid_summary[0].index(3):
                    valid_summary[0][valid_summary[0].index(3)] = 1
                else:
                    valid_summary[0][-1] = 1

                    # tensor, gpu
                valid_doc = torch.tensor(valid_doc)
                valid_summary = torch.tensor(valid_summary)
                valid_doc = valid_doc.to(device)
                valid_summary = valid_summary.to(device)

                # Evaluation
                outputs1 = model(valid_doc, labels=valid_summary)

                # Save Best Performance Model
                if outputs1.loss < best_valid_loss:
                    best_valid_loss = outputs1.loss
                    print(best_valid_loss)
                    torch.save(model.state_dict(), "Summarization_model.pt")

    model.load_state_dict(torch.load("Summarization_model.pt"))
    
    return model
