MAX_DOC_LENGTH = 500
padding_ID = 1
unknown_ID = 0    
def encode_data(data_path, vocab_path):
    with open(vocab_path) as f:
        vocab = dict([(word, word_ID + 2) for word_ID, word in enumerate(f.read().splitlines())])
    with open(data_path) as f:
        documents = [(line.split("<fff>")[0], line.split("<fff>")[1], line.split("<fff>")[2]) for line in f.read().splitlines()]
    encoded_data = []
    for document in documents:
        label, doc_id, text = document
        words = text.split()[:MAX_DOC_LENGTH]
        sentence_length = len(words)
        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(unknown_ID))
        if len(word) < MAX_DOC_LENGTH:
            num_padding = MAX_DOC_LENGTH - len(word)
            for _ in range(num_padding):
                encoded_text.append(str(padding_ID))
        encoded_data.append(str(label) + "<fff>" + str(doc_id) + "<fff>" + str(sentence_length) + "<fff>" + " ".join(encoded_text))

    dir_name = "/".join(data_path.split("/")[:-1])
    file_name = "-".join(data_path.split("/")[-1].split("-")[:-1]) + "-encoded.txt"
    with open(dir_name  + "/" + file_name, "w") as f:
        f.write("\n".join(encoded_data))
train_data_path = 'D:/DSLab/DSLab_Training_Phase1/Dataset/20news-bydate/20news-train-raw.txt'
test_data_path = 'D:/DSLab/DSLab_Training_Phase1/Dataset/20news-bydate/20news-test-raw.txt'
vocab_path = 'D:/DSLab/DSLab_Training_Phase1/Dataset/20news-bydate/vocab-raw.txt'

encode_data(train_data_path, vocab_path)
encode_data(test_data_path, vocab_path)
