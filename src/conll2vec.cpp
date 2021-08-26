#include "command_line_parameters_defs.h"
#include "simple_profiler.h"
#include "fit_parus.h"
#include "vocabs_builder.h"
#include "original_word2vec_vocabulary.h"
#include "learning_example_provider.h"
#include "trainer.h"
#include "sim_estimator.h"
#include "selftest_ru.h"
#include "unpnizer.h"
#include "add_punct.h"
#include "add_toks.h"
#include "balance.h"
#include "sseval.h"
#include "vectors_model.h"

#include <memory>
#include <string>
#include <iostream>
#include <thread>



// создание объекта, отвечающего за измерение семантической близости между словами
std::shared_ptr<SimilarityEstimator> create_sim_estimator(const CommandLineParametersDefs& cmdLineParams)
{
  std::shared_ptr<SimilarityEstimator> sim_estimator = std::make_shared<SimilarityEstimator>( cmdLineParams.getAsInt("-size_d"),
                                                                                              cmdLineParams.getAsInt("-size_a"),
                                                                                              cmdLineParams.getAsInt("-size_g"),
                                                                                              cmdLineParams.getAsFloat("-a_ratio") );
  if ( !sim_estimator->load_model(cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt")) )
    return nullptr;
  return sim_estimator;
}



int main(int argc, char **argv)
{
  // выполняем разбор параметров командной строки
  CommandLineParametersDefs cmdLineParams;
  cmdLineParams.parse(argc, argv);
  cmdLineParams.dbg_cout();

  // определяемся с поставленной задачей
  if ( !cmdLineParams.isDefined("-task") )
  {
    std::cerr << "Task parameter is not defined." << std::endl;
    std::cerr << "Alternatives:" << std::endl
              << "  -task fit         -- conll file transformation" << std::endl
              << "  -task vocab       -- vocabs building" << std::endl
              << "  -task train       -- model training" << std::endl
              << "  -task punct       -- add punctuation to model" << std::endl
              << "  -task sim         -- similarity test" << std::endl
              << "  -task selftest_ru -- model self-test for russian" << std::endl
              << "  -task unPNize     -- merge common & proper names models" << std::endl
              << "  -task toks        -- add tokens to model" << std::endl
              << "  -task toks_train  -- train tokens model" << std::endl
              << "  -task toks_gramm  -- train grammatical embeddings and append them to model" << std::endl
              << "  -task balance     -- balance model dep/assoc ratio" << std::endl
              << "  -task sub         -- extract sub-model (for dimensions range)" << std::endl
              << "  -task fsim        -- calc similarity measure for word pairs in file" << std::endl
              << "  -task normalize   -- normalize vectors length" << std::endl
              << "  -task sseval      -- subsampling value estimation" << std::endl;
    return -1;
  }
  auto&& task = cmdLineParams.getAsString("-task");

  // если поставлена задача преобразования conll-файла
  if (task == "fit")
  {
    FitParus fitter;
    fitter.run( cmdLineParams.getAsString("-fit_input"), cmdLineParams.getAsString("-train") );
    return 0;
  }

  // если поставлена задача построения словарей
  if (task == "vocab")
  {
    VocabsBuilder vb;
    bool succ = vb.build_vocabs( cmdLineParams.getAsString("-train"),
                                 cmdLineParams.getAsString("-vocab_m"), cmdLineParams.getAsString("-vocab_p"), cmdLineParams.getAsString("-vocab_t"),
                                 cmdLineParams.getAsString("-tl_map"), cmdLineParams.getAsString("-vocab_o"), cmdLineParams.getAsString("-vocab_d"),
                                 cmdLineParams.getAsInt("-min-count_m"), cmdLineParams.getAsInt("-min-count_p"), cmdLineParams.getAsInt("-min-count_t"),
                                 cmdLineParams.getAsInt("-min-count_o"), cmdLineParams.getAsInt("-min-count_d"),
                                 cmdLineParams.getAsInt("-col_ctx_d") - 1, (cmdLineParams.getAsInt("-use_deprel") == 1), (cmdLineParams.getAsInt("-exclude_nums") == 1),
                                 cmdLineParams.getAsInt("-max_oov_sfx")
                               );
    return ( succ ? 0 : -1 );
  }

  // если поставлена задача обучения модели
  if (task == "train")
  {
    if ( !cmdLineParams.isDefined("-train") )
    {
      std::cerr << "Trainset is not defined." << std::endl;
      return -1;
    }
    if ( !cmdLineParams.isDefined("-model") )
    {
      std::cerr << "-model parameter must be defined." << std::endl;
      return -1;
    }
    if ( !cmdLineParams.isDefined("-vocab_m") && !cmdLineParams.isDefined("-vocab_p") )
    {
      std::cerr << "-vocab_m or -vocab_p parameter must be defined." << std::endl;
      return -1;
    }
    if ( cmdLineParams.isDefined("-vocab_m") && cmdLineParams.isDefined("-vocab_p") )
    {
      std::cerr << "-vocab_p parameter will be ignored." << std::endl;
    }
    if ( cmdLineParams.isDefined("-vocab_p") && !cmdLineParams.isDefined("-restore") )
    {
      std::cerr << "-restore parameter must be defined (when -vocab_p is defined)." << std::endl;
      return -1;
    }
    if ( cmdLineParams.getAsInt("-size_d") > 0 && !cmdLineParams.isDefined("-vocab_d") ) // устанавливая -size_d 0, можно строить только ассоциативную модель
    {
      std::cerr << "-vocab_d parameter must be defined." << std::endl;
      return -1;
    }
    if ( cmdLineParams.getAsInt("-size_a") > 0 && !cmdLineParams.isDefined("-vocab_a") ) // устанавливая -size_a 0, можно строить только синтаксическую модель
    {
      std::cerr << "-vocab_a parameter must be defined." << std::endl;
      return -1;
    }

    SimpleProfiler global_profiler;

    // загрузка словарей
    bool needLoadMainVocab = cmdLineParams.isDefined("-vocab_m");
    bool needLoadProperVocab = !needLoadMainVocab;
    bool needLoadDepCtxVocab = (cmdLineParams.getAsInt("-size_d") > 0);
    bool needLoadAssocCtxVocab = (cmdLineParams.getAsInt("-size_a") > 0);
    std::shared_ptr< OriginalWord2VecVocabulary > v_main, v_proper, v_dep_ctx, v_assoc_ctx;
    if (needLoadMainVocab)
    {
      v_main = std::make_shared<OriginalWord2VecVocabulary>();
      if ( !v_main->load( cmdLineParams.getAsString("-vocab_m") ) )
        return -1;
    }
    if (needLoadProperVocab)
    {
      v_proper = std::make_shared<OriginalWord2VecVocabulary>();
      if ( !v_proper->load( cmdLineParams.getAsString("-vocab_p") ) )
        return -1;
    }
    if (needLoadDepCtxVocab)
    {
      v_dep_ctx = std::make_shared<OriginalWord2VecVocabulary>();
      if ( !v_dep_ctx->load( cmdLineParams.getAsString("-vocab_d") ) )
        return -1;
    }
    if (needLoadAssocCtxVocab)
    {
      v_assoc_ctx = std::make_shared<OriginalWord2VecVocabulary>();
      v_assoc_ctx->init_stoplist("stopwords.assoc");
      if ( !v_assoc_ctx->load( cmdLineParams.getAsString("-vocab_a") ) )
        return -1;
    }

    // создание поставщика обучающих примеров
    // к моменту создания "поставщика обучающих примеров" словарь должен быть загружен (в частности, используется cn_sum())
    std::shared_ptr< LearningExampleProvider> lep = std::make_shared< LearningExampleProvider > ( cmdLineParams.getAsString("-train"),
                                                                                                  cmdLineParams.getAsInt("-threads"),
                                                                                                  (needLoadMainVocab ? v_main : v_proper ),
                                                                                                  needLoadProperVocab,
                                                                                                  v_dep_ctx, v_assoc_ctx,
                                                                                                  2,
                                                                                                  cmdLineParams.getAsInt("-col_ctx_d") - 1,
                                                                                                  (cmdLineParams.getAsInt("-use_deprel") == 1),
                                                                                                  false, 0,
                                                                                                  cmdLineParams.getAsFloat("-sample_w"),
                                                                                                  cmdLineParams.getAsFloat("-sample_d"),
                                                                                                  cmdLineParams.getAsFloat("-sample_a")
                                                                                                );

    // создаем объект, организующий обучение
    Trainer trainer( lep, (needLoadMainVocab ? v_main : v_proper ), needLoadProperVocab,
                     v_dep_ctx, v_assoc_ctx,
                     cmdLineParams.getAsInt("-size_d"),
                     cmdLineParams.getAsInt("-size_a"),
                     0,
                     cmdLineParams.getAsInt("-iter"),
                     cmdLineParams.getAsFloat("-alpha"),
                     cmdLineParams.getAsInt("-negative"),
                     cmdLineParams.getAsInt("-threads") );

    // инициализация нейросети
    if (needLoadMainVocab)
    {
      trainer.create_net();
      trainer.init_net();
    }
    else
    {
      trainer.create_net();
      trainer.init_net();  // инициализация левой матрицы случайными значениями (для словаря собственных имен)
      VectorsModel vm;
      if ( !vm.load(cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt")) )
        return -1;
      trainer.restore_assoc_by_model(vm);
      trainer.restore( cmdLineParams.getAsString("-restore"), false, true );
    }

    // запускаем потоки, осуществляющие обучение
    size_t threads_count = cmdLineParams.getAsInt("-threads");
    std::vector<std::thread> threads_vec;
    threads_vec.reserve(threads_count);
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec.emplace_back(&Trainer::train_entry_point, &trainer, i);
    // ждем завершения обучения
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec[i].join();

    // сохраняем вычисленные вектора в файл
    if (needLoadMainVocab)
    {
      if (cmdLineParams.isDefined("-model"))
        trainer.saveEmbeddings( cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt") );
      if (cmdLineParams.isDefined("-backup"))
        trainer.backup( cmdLineParams.getAsString("-backup"), false, true );
    }
    else
    {
      v_proper->suffixize("_PN");
      if (cmdLineParams.isDefined("-model"))
        trainer.appendEmbeddings( cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt") );
    }

    return 0;
  } // if task == train

  // если поставлена задача добавления в модель знаков пунктуации
  if (task == "punct")
  {
    AddPunct::run(cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt"));
    return 0;
  } // if task == punct

  // если поставлена задача оценки близости значений (в интерактивном режиме)
  if (task == "sim")
  {
    auto sim_estimator = create_sim_estimator(cmdLineParams);
    if (!sim_estimator)
      return -1;
    sim_estimator->run();
    return 0;
  } // if task == sim

  // если поставлена задача самодиагностики (язык: русский)
  if (task == "selftest_ru")
  {
    auto sim_estimator = create_sim_estimator(cmdLineParams);
    if (!sim_estimator)
      return -1;
    SelfTest_ru st(sim_estimator, (cmdLineParams.getAsInt("-st_yo")==1));
    st.run(false);
    return 0;
  } // if task == sefltest_ru

  // если поставлена задача сведения моделей для нарицательных и собственных имен
  if (task == "unPNize")
  {
    std::shared_ptr< OriginalWord2VecVocabulary > v_main, v_proper;
    v_main = std::make_shared<OriginalWord2VecVocabulary>();
    if ( !v_main->load( cmdLineParams.getAsString("-vocab_m") ) )
      return -1;
    v_proper = std::make_shared<OriginalWord2VecVocabulary>();
    if ( !v_proper->load( cmdLineParams.getAsString("-vocab_p") ) )
      return -1;
    Unpnizer::run(v_main, v_proper, cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt"));
    return 0;
  } // if task == unPNize

  // если поставлена задача добавления токенов в модель
  if (task == "toks")
  {
    AddToks::run(cmdLineParams.getAsString("-model"), cmdLineParams.getAsString("-tl_map"), (cmdLineParams.getAsString("-model_fmt") == "txt"));
    return 0;
  } // if task == toks

  // если поставлена задача доучивания модели токенов
  if (task == "toks_train")
  {
    if ( !cmdLineParams.isDefined("-train") )
    {
      std::cerr << "Trainset is not defined." << std::endl;
      return -1;
    }
    if ( !cmdLineParams.isDefined("-model") )
    {
      std::cerr << "-model parameter must be defined." << std::endl;
      return -1;
    }
    if ( !cmdLineParams.isDefined("-vocab_t") )
    {
      std::cerr << "-vocab_t parameter must be defined." << std::endl;
      return -1;
    }
    if ( !cmdLineParams.isDefined("-restore") )
    {
      std::cerr << "-restore parameter must be defined." << std::endl;
      return -1;
    }
    if ( !cmdLineParams.isDefined("-vocab_d") && cmdLineParams.getAsInt("-size_d") > 0 ) // устанавливая -size_d 0, можно строить только ассоциативную модель
    {
      std::cerr << "-vocab_d parameter must be defined." << std::endl;
      return -1;
    }
    if ( !cmdLineParams.isDefined("-vocab_a") && cmdLineParams.getAsInt("-size_a") > 0 ) // устанавливая -size_a 0, можно строить только синтаксическую модель
    {
      std::cerr << "-vocab_a parameter must be defined." << std::endl;
      return -1;
    }

    SimpleProfiler global_profiler;

    // загрузка словарей
    bool needLoadDepCtxVocab = (cmdLineParams.getAsInt("-size_d") > 0);
    bool needLoadAssocCtxVocab = (cmdLineParams.getAsInt("-size_a") > 0);
    std::shared_ptr< OriginalWord2VecVocabulary > v_toks, v_dep_ctx, v_assoc_ctx;
    v_toks = std::make_shared<OriginalWord2VecVocabulary>();
    if ( !v_toks->load( cmdLineParams.getAsString("-vocab_t") ) )
      return -1;
    if (needLoadDepCtxVocab)
    {
      v_dep_ctx = std::make_shared<OriginalWord2VecVocabulary>();
      if ( !v_dep_ctx->load( cmdLineParams.getAsString("-vocab_d") ) )
        return -1;
    }
    if (needLoadAssocCtxVocab)
    {
      v_assoc_ctx = std::make_shared<OriginalWord2VecVocabulary>();
      v_assoc_ctx->init_stoplist("stopwords.assoc");
      if ( !v_assoc_ctx->load( cmdLineParams.getAsString("-vocab_a") ) )
        return -1;
    }

    // загрузка векторной модели
    VectorsModel vm;
    if ( !vm.load(cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt")) )
      return -1;

    // создание поставщика обучающих примеров
    // к моменту создания "поставщика обучающих примеров" словарь должен быть загружен (в частности, используется cn_sum())
    std::shared_ptr< LearningExampleProvider> lep = std::make_shared< LearningExampleProvider > ( cmdLineParams.getAsString("-train"),
                                                                                                  cmdLineParams.getAsInt("-threads"),
                                                                                                  v_toks, false, v_dep_ctx, v_assoc_ctx,
                                                                                                  1,
                                                                                                  cmdLineParams.getAsInt("-col_ctx_d") - 1,
                                                                                                  (cmdLineParams.getAsInt("-use_deprel") == 1),
                                                                                                  false, 0,
                                                                                                  cmdLineParams.getAsFloat("-sample_w"),
                                                                                                  cmdLineParams.getAsFloat("-sample_d"),
                                                                                                  cmdLineParams.getAsFloat("-sample_a")
                                                                                                );

    // создаем объект, организующий обучение
    Trainer trainer( lep, v_toks, false,
                     v_dep_ctx, v_assoc_ctx,
                     cmdLineParams.getAsInt("-size_d"),
                     cmdLineParams.getAsInt("-size_a"),
                     0,
                     cmdLineParams.getAsInt("-iter"),
                     cmdLineParams.getAsFloat("-alpha"),
                     cmdLineParams.getAsInt("-negative"),
                     cmdLineParams.getAsInt("-threads") );

    // инициализация нейросети
    trainer.create_net();
    trainer.init_net();  // начальная инициализация левой матрицы случайными значениями
    trainer.restore_left_matrix_by_model(vm);  // перенос векторых представлений из загруженной модели в левую матрицу
    trainer.restore( cmdLineParams.getAsString("-restore"), false, true );

    // запускаем потоки, осуществляющие обучение
    size_t threads_count = cmdLineParams.getAsInt("-threads");
    std::vector<std::thread> threads_vec;
    threads_vec.reserve(threads_count);
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec.emplace_back(&Trainer::train_entry_point, &trainer, i);
    // ждем завершения обучения
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec[i].join();

    // сохраняем вычисленные вектора в файл
    trainer.saveEmbeddings( cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt") );
    return 0;
  } // if task == toks_train

  // если поставлена задача добавления в модель грамматических эмбеддингов
  if (task == "toks_gramm")
  {
    // загрузим модель
    VectorsModel vm;
    if ( !vm.load(cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt")) )
      return -1;
    // загрузим словарь токенов
    std::shared_ptr< OriginalWord2VecVocabulary > v_toks = std::make_shared<OriginalWord2VecVocabulary>();
    if ( !v_toks->load( cmdLineParams.getAsString("-vocab_t") ) )
      return -1;
    // если работаем со словарем суффиксов, до догружаем его в словарь токенов
    std::string oovv = cmdLineParams.getAsString("-vocab_o");
    if (!oovv.empty())
    {
      if ( !v_toks->load(oovv) )
        return -1;
    }
    // создание поставщика обучающих примеров
    std::shared_ptr< LearningExampleProvider> lep = std::make_shared< LearningExampleProvider > ( cmdLineParams.getAsString("-train"),
                                                                                                  cmdLineParams.getAsInt("-threads"),
                                                                                                  v_toks, false, nullptr, nullptr,
                                                                                                  1,
                                                                                                  0, false, !oovv.empty(), cmdLineParams.getAsInt("-max_oov_sfx"),
                                                                                                  cmdLineParams.getAsFloat("-sample_w"), 0, 0
                                                                                                );
    // создаем объект, организующий обучение
    Trainer trainer( lep, v_toks, false, nullptr, nullptr,
                     cmdLineParams.getAsInt("-size_d"),
                     cmdLineParams.getAsInt("-size_a"),
                     cmdLineParams.getAsInt("-size_g"),
                     cmdLineParams.getAsInt("-iter"),
                     cmdLineParams.getAsFloat("-alpha"),
                     cmdLineParams.getAsInt("-negative"),
                     cmdLineParams.getAsInt("-threads") );

    // инициализация нейросети
    trainer.create_and_init_gramm_net();

    size_t threads_count = cmdLineParams.getAsInt("-threads");
    // запускаем потоки, осуществляющие обучение
    std::vector<std::thread> threads_vec;
    threads_vec.reserve(threads_count);
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec.emplace_back(&Trainer::train_entry_point__gramm, &trainer, i);
    // ждем завершения обучения
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec[i].join();

    // сохраняем вычисленные вектора в файл
    trainer.saveGrammaticalEmbeddings( vm, cmdLineParams.getAsFloat("-g_ratio"), oovv, cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt") );

    return 0;
  } // if task == toks_gramm

  // если поставлена задача балансировки модели (изменения весового соотношения dep и assoc частей)
  if (task == "balance")
  {
    Balancer::run(cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt"),
                  cmdLineParams.getAsInt("-size_d"), cmdLineParams.getAsInt("-size_a"), cmdLineParams.getAsFloat("-a_ratio"));
    return 0;
  } // if task == balance

  // если поставлена задача извлечения подмодели
  if (task == "sub")
  {
    if ( !cmdLineParams.isDefined("-sub_l") || !cmdLineParams.isDefined("-sub_r") )
    {
      std::cerr << "-sub_l and -sub_r parameters must be defined." << std::endl;
      return -1;
    }
    size_t lb = cmdLineParams.getAsInt("-sub_l");
    size_t rb = cmdLineParams.getAsInt("-sub_r");
    std::string model_fn = cmdLineParams.getAsString("-model");
    bool useTxtFmt = (cmdLineParams.getAsString("-model_fmt") == "txt");
    VectorsModel vm;
    if ( !vm.load(model_fn, useTxtFmt) )
      return -1;
    FILE *fo = fopen(model_fn.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", vm.words_count, rb-lb);
    for (size_t a = 0; a < vm.vocab.size(); ++a)
      VectorsModel::write_embedding_slice(fo, useTxtFmt, vm.vocab[a], &vm.embeddings[a * vm.emb_size], lb, rb);
    fclose(fo);
    return 0;
  } // if task == sub

  // если поставлена задача оценки близости значений (в пакетном режиме)
  if (task == "fsim")
  {
    auto sim_estimator = create_sim_estimator(cmdLineParams);
    if (!sim_estimator)
      return -1;
    sim_estimator->run_for_file(cmdLineParams.getAsString("-fsim_file"), cmdLineParams.getAsString("-fsim_fmt"));
    return 0;
  } // if task == fsim

  // если поставлена задача нормализации векторов
  if (task == "normalize")
  {
    std::string model_fn = cmdLineParams.getAsString("-model");
    bool useTxtFmt = (cmdLineParams.getAsString("-model_fmt") == "txt");
    VectorsModel vm;
    if ( !vm.load(model_fn, useTxtFmt, true) )
      return -1;
    vm.save(model_fn, useTxtFmt);
    return 0;
  } // if task == normalize

  // если поставлена задача оценки вариантов subsampling'а для данного словаря
  if (task == "sseval")
  {
    SsEval::run(cmdLineParams.getAsString("-eval_vocab"));
    return 0;
  } // if task == sseval

  return -1;
}
