@echo off

SETLOCAL
SET SIZE_DEP=75
SET SIZE_ASSOC=25
SET TRAIN_FN=parus_first_10m_lines.conll
SET COL_EMB=3
SET COL_CTX_D=3
SET USE_DEPREL=1
SET MODEL_FN=vectors.bin
SET VOC_M=main.vocab
SET VOC_P=proper.vocab
SET VOC_D=ctx_dep.vocab
SET VOC_A=ctx_assoc.vocab
SET THREADS=8


if not exist conll2vec.exe (
  echo.
  echo "MAKING BINARIES"
  nmake -f makefile.msvc
  copy data\stopwords.assoc .
)

if not exist %TRAIN_FN% (
  echo.
  echo TRAINSET EXTRACTION AND FITTING
  echo   please wait...
  cscript //NoLogo helpers\demo.cmd.unzip.vbs %cd% %cd%\data\parus_first_10m_lines.conll.zip
  rename parus_first_10m_lines.conll pre_fit.conll
  conll2vec -task fit -fit_input pre_fit.conll -train %TRAIN_FN%
  del /f pre_fit.conll
)

echo.
echo BUILDING VOCABULARIES
conll2vec -task vocab -train %TRAIN_FN% -col_emb %COL_EMB% -col_ctx_d %COL_CTX_D% -vocab_m %VOC_M% -vocab_p %VOC_P% -vocab_d %VOC_D% -vocab_a %VOC_A% -min-count_m 70 -min-count_p 100 -min-count_d 20 -min-count_a 20 -use_deprel %USE_DEPREL%

echo.
echo TRAINING EMBEDDINGS -- MAIN
conll2vec -task train -train %TRAIN_FN% -col_emb %COL_EMB% -col_ctx_d %COL_CTX_D% -use_deprel %USE_DEPREL% -vocab_m %VOC_M% -backup backup.data -vocab_d %VOC_D% -vocab_a %VOC_M% -model %MODEL_FN% -size_d %SIZE_DEP% -size_a %SIZE_ASSOC% -negative 4 -iter 10 -threads %THREADS%

echo.
echo TRAINING EMBEDDINGS -- PROPER
conll2vec -task train -train %TRAIN_FN% -col_emb %COL_EMB% -col_ctx_d %COL_CTX_D% -use_deprel %USE_DEPREL% -vocab_p %VOC_P% -restore backup.data -vocab_d %VOC_D% -vocab_a %VOC_M% -model %MODEL_FN% -size_d %SIZE_DEP% -size_a %SIZE_ASSOC% -negative 4 -iter 5 -threads %THREADS%

echo.
echo RUN SIMILARITY METER
conll2vec -task sim -model %MODEL_FN% -model_fmt bin -size_d %SIZE_DEP% -size_a %SIZE_ASSOC%
