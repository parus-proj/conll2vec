SIZE_DEP=60
SIZE_ASSOC=40
SIZE_GRAMM=0
TRAIN_FN=parus_first_10m_lines.conll
COL_CTX_D=3
USE_DEPREL=1
MODEL_FN=vectors.c2v
VOC_M=main.vocab
VOC_T=tokens.vocab
VOC_D=ctx_dep.vocab
THREADS=8

echo "MAKING BINARIES"
make
cp ./data/stopwords.assoc ./

echo ""
echo "TRAINSET EXTRACTION AND FITTING"
gzip --decompress --stdout ./data/parus_first_10m_lines.conll.zip | ./conll2vec -task fit -fit_input stdin -train $TRAIN_FN

echo ""
echo "BUILDING VOCABULARIES"
./conll2vec -task vocab -train $TRAIN_FN \
            -vocab_l $VOC_M -vocab_t $VOC_T -vocab_d $VOC_D -col_ctx_d $COL_CTX_D -use_deprel $USE_DEPREL \
            -min-count_m 70 -min-count_t 50 -min-count_d 20 -exclude_nums 1

echo ""
echo "TRAINING EMBEDDINGS -- MAIN"
./conll2vec -task train -train $TRAIN_FN \
            -vocab_l $VOC_M -backup backup.data -vocab_d $VOC_D -col_ctx_d $COL_CTX_D -use_deprel $USE_DEPREL -model $MODEL_FN \
            -sample_w 1e-4 -sample_d 1e-4 -sample_a 1e-4 \
            -size_d $SIZE_DEP -size_a $SIZE_ASSOC -negative_d 4 -negative_a 3 -iter 10 -threads $THREADS

echo ""
echo "RUN SIMILARITY METER"
./conll2vec -task sim -model $MODEL_FN

