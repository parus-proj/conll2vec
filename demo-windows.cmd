@echo off

SETLOCAL
SET SIZE_DEP=60
SET SIZE_ASSOC=40
SET SIZE_GRAMM=0
SET TRAIN_FN=parus_first_10m_lines.conll
SET COL_CTX_D=3
SET USE_DEPREL=1
SET MODEL_FN=vectors.c2v
SET VOC_M=main.vocab
SET VOC_D=ctx_dep.vocab
SET VOC_T=tokens.vocab
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
conll2vec -task vocab -train %TRAIN_FN% -vocab_l %VOC_M% -vocab_t %VOC_T% -vocab_d %VOC_D% -min-count_m 70 -min-count_t 50 -min-count_d 20 -use_deprel %USE_DEPREL% -col_ctx_d %COL_CTX_D% -exclude_nums 1

echo.
echo TRAINING EMBEDDINGS -- MAIN
conll2vec -task train -train %TRAIN_FN% -vocab_l %VOC_M% -backup backup.data -vocab_d %VOC_D% -model %MODEL_FN% -size_d %SIZE_DEP% -size_a %SIZE_ASSOC% -sample_w 1e-4 -sample_d 1e-4 -sample_a 1e-4 -negative_d 4 -negative_a 3 -iter 10 -col_ctx_d %COL_CTX_D% -use_deprel %USE_DEPREL% -threads %THREADS%

echo.
echo RUN SIMILARITY METER
conll2vec -task sim -model %MODEL_FN%
