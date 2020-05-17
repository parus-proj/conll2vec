SIZE_DEP=80
SIZE_ASSOC=20
TRAIN_FN=parus_first_10m_lines.conll
COL_EMB=3
COL_CTX_D=3
USE_DEPREL=1
MODEL_FN=vectors.bin
VOC_M=main.vocab
VOC_P=proper.vocab
VOC_D=ctx_dep.vocab
VOC_A=ctx_assoc.vocab
THREADS=8

echo "MAKING BINARIES"
make
cp ./data/stopwords.assoc ./

echo ""
echo "TRAINSET EXTRACTION AND FITTING"
gzip --decompress --stdout ./data/parus_first_10m_lines.conll.zip | ./conll2vec -task fit -fit_input stdin -train $TRAIN_FN

echo ""
echo "BUILDING VOCABULARIES"
./conll2vec -task vocab -train $TRAIN_FN -col_emb $COL_EMB -col_ctx_d $COL_CTX_D -vocab_m $VOC_M -vocab_p $VOC_P -vocab_d $VOC_D -vocab_a $VOC_A -min-count_m 70 -min-count_p 100 -min-count_d 20 -min-count_a 20 -use_deprel $USE_DEPREL

echo ""
echo "TRAINING EMBEDDINGS -- MAIN"
./conll2vec -task train -train $TRAIN_FN -col_emb $COL_EMB -col_ctx_d $COL_CTX_D -use_deprel $USE_DEPREL -vocab_m $VOC_M -backup backup.data -vocab_d $VOC_D -vocab_a $VOC_A -model $MODEL_FN -size_d $SIZE_DEP -size_a $SIZE_ASSOC -negative 5 -iter 10 -threads $THREADS

echo ""
echo "TRAINING EMBEDDINGS -- PROPER"
./conll2vec -task train -train $TRAIN_FN -col_emb $COL_EMB -col_ctx_d $COL_CTX_D -use_deprel $USE_DEPREL -vocab_p $VOC_P -restore backup.data -vocab_d $VOC_D -vocab_a $VOC_A -model $MODEL_FN -size_d $SIZE_DEP -size_a $SIZE_ASSOC -negative 5 -iter 5 -threads $THREADS

echo ""
echo "RUN SIMILARITY METER"
./conll2vec -task sim -model $MODEL_FN -model_fmt bin -size_d $SIZE_DEP -size_a $SIZE_ASSOC

