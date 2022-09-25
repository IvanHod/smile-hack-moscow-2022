def _fit_tfidf(self, sequences_matrix_path: str) -> TfidfVectorizer:
    data = pd.read_csv(sequences_matrix_path, sep='\t')
    data['seq'] = data['SESSIONS_SEQUENCES'].apply(
        lambda seq_list: seq_list.replace("[", "").replace("]", "").replace("', '", " ").replace("'", " ").strip()
    )

    corpus = data['seq'].values
    count_tf_idf = TfidfVectorizer(max_features=1000)
    tf_idf = count_tf_idf.fit(corpus)

    return tf_idf


def _fit_w2v(self, sequences_matrix_path: str) -> Word2Vec:
    data = pd.read_csv(sequences_matrix_path, sep='\t')
    display(data)
    list_of_lists = list(map(lambda v: eval(v), data['SESSIONS_SEQUENCES'].to_list()))

    w2v_model = Word2Vec(min_count=3, vector_size=1000)
    w2v_model.build_vocab(list_of_lists)
    w2v_model.train(list_of_lists, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)

    return w2v_model