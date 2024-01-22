#include "command_line_parameters_defs.h"
#include "simple_profiler.h"
#include "fit_parus.h"
#include "vocabs_builder.h"
#include "original_word2vec_vocabulary.h"
#include "mwe_vocabulary.h"
#include "learning_example_provider.h"
#include "trainer.h"
#include "sim_estimator.h"
#include "selftest_ru.h"
#include "add_punct.h"
#include "add_toks.h"
#include "balance.h"
#include "sseval.h"
#include "vectors_model.h"
#include "ra_vocab.h"
#include "categoroid_vocab.h"
#include "model_splitter.h"
#include "make_rue_embeddings.h"
#include "extract_related.h"

#include <memory>
#include <string>
#include <iostream>
#include <thread>



// создание объекта, отвечающего за измерение семантической близости между словами
std::shared_ptr<SimilarityEstimator> create_sim_estimator(const CommandLineParametersDefs& cmdLineParams)
{
  if ( !cmdLineParams.isDefined("-model") )
  {
    std::cerr << "-model parameter must be defined." << std::endl;
    return nullptr;
  }
  std::shared_ptr<SimilarityEstimator> sim_estimator = std::make_shared<SimilarityEstimator>( cmdLineParams.getAsFloat("-a_ratio") );
  if ( !sim_estimator->load_model(cmdLineParams.getAsString("-model")) )
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
              << "  -task train       -- lemmas model training" << std::endl
              << "  -task sim         -- similarity test" << std::endl
              << "  -task selftest_ru -- model self-test for russian" << std::endl
              << "  -task punct       -- add punctuation to model" << std::endl
              << "  -task toks        -- add tokens to model" << std::endl
              << "  -task toks_train  -- train tokens model" << std::endl
              << "  -task toks_gramm  -- train grammatical embeddings and append them to model" << std::endl
              << "  -task import      -- import from word2vec model" << std::endl
              << "  -task export      -- export to word2vec model" << std::endl
              << "  -task balance     -- balance model dep/assoc ratio" << std::endl
              << "  -task sub         -- extract sub-model (for dimensions range)" << std::endl
              << "  -task fsim        -- calc similarity measure for word pairs in file" << std::endl
              << "  -task sseval      -- subsampling value estimation" << std::endl
              << "  -task extract     -- extract related pairs from model" << std::endl
              << "  -task emerge      -- merge extracted related pairs" << std::endl
              << "  -task spl_m       -- split model (stem, suffix)" << std::endl
              << "  -task rue         -- prepare RUE embeddings" << std::endl;
    return -1;
  }
  auto&& task = cmdLineParams.getAsString("-task");

  // если поставлена задача преобразования conll-файла
  if (task == "fit")
  {
    FitParus fitter( (cmdLineParams.getAsInt("-exclude_nums") == 1) );
    fitter.run( cmdLineParams.getAsString("-fit_input"), cmdLineParams.getAsString("-train") );
    return 0;
  }

  // если поставлена задача построения словарей
  if (task == "vocab")
  {
    SimpleProfiler global_profiler;
    VocabsBuilder vb;
    bool succ = vb.build_vocabs( cmdLineParams.getAsString("-train"),
                                 cmdLineParams.getAsString("-vocab_l"), cmdLineParams.getAsString("-vocab_t"),
                                 cmdLineParams.getAsString("-tl_map"), cmdLineParams.getAsString("-vocab_o"), cmdLineParams.getAsString("-vocab_d"),
                                 cmdLineParams.getAsInt("-min-count_l"), cmdLineParams.getAsInt("-min-count_t"),
                                 cmdLineParams.getAsInt("-min-count_o"), cmdLineParams.getAsInt("-min-count_d"),
                                 cmdLineParams.getAsInt("-col_ctx_d") - 1, (cmdLineParams.getAsInt("-use_deprel") == 1),
                                 cmdLineParams.getAsInt("-max_oov_sfx"), cmdLineParams.getAsString("-ca_vocab"), cmdLineParams.getAsString("-vocab_e"),
                                 cmdLineParams.getAsInt("-threads")
                               );
    std::cout << '\n' << "Vocab building: "; // profiler str prefix
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
    if ( !cmdLineParams.isDefined("-vocab_l") )
    {
      std::cerr << "-vocab_l parameter must be defined." << std::endl;
      return -1;
    }
    if ( cmdLineParams.getAsInt("-size_d") > 0 && !cmdLineParams.isDefined("-vocab_d") ) // устанавливая -size_d 0, можно строить только ассоциативную модель
    {
      std::cerr << "-vocab_d parameter must be defined." << std::endl;
      return -1;
    }

    SimpleProfiler global_profiler;

    // загрузка словарей
    std::shared_ptr< OriginalWord2VecVocabulary > v_main, v_dep_ctx, v_assoc_ctx;
    std::shared_ptr< MweVocabulary > v_mwe;
    v_main = std::make_shared<OriginalWord2VecVocabulary>();
    if ( !v_main->load( cmdLineParams.getAsString("-vocab_l") ) )
      return -1;
    v_mwe = std::make_shared<MweVocabulary>( );
    if ( !v_mwe->load( cmdLineParams.getAsString("-vocab_e"), v_main ) )
      return -1;
    bool needLoadDepCtxVocab = (cmdLineParams.getAsInt("-size_d") > 0);
    bool needLoadAssocCtxVocab = (cmdLineParams.getAsInt("-size_a") > 0);
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
      if ( !v_assoc_ctx->load( cmdLineParams.getAsString("-vocab_l") ) )
        return -1;
    }

    std::shared_ptr< ExternalVocabsManager > ext_vocab_manager;
    if ( cmdLineParams.isDefined("-vocabs_tab"))
    {
      ext_vocab_manager = std::make_shared<ExternalVocabsManager>();
      if ( !ext_vocab_manager->load( cmdLineParams.getAsString("-vocabs_tab") ) )
        return -1;
      if ( !ext_vocab_manager->load_vocabs(v_main) )
        return -1;
    }

    std::shared_ptr< ReliableAssociativesVocabulary > ra_vocab;
    if ( cmdLineParams.isDefined("-rr_vocab"))
    {
      ra_vocab = std::make_shared<ReliableAssociativesVocabulary>();
      if ( !ra_vocab->load( cmdLineParams.getAsString("-rr_vocab"), v_main ) )
        return -1;
    }

    // создание поставщика обучающих примеров
    // к моменту создания "поставщика обучающих примеров" словарь должен быть загружен (в частности, используется cn_sum())
    std::shared_ptr< LearningExampleProvider> lep = std::make_shared< LearningExampleProvider > ( cmdLineParams,
                                                                                                  v_main, false, v_dep_ctx, v_assoc_ctx, v_mwe,
                                                                                                  2, false, 0,
                                                                                                  ra_vocab, ext_vocab_manager
                                                                                                );

    // создаем объект, организующий обучение
    Trainer trainer( lep, v_main, false, v_dep_ctx, v_assoc_ctx,
                     cmdLineParams.getAsInt("-size_d"),
                     cmdLineParams.getAsInt("-size_a"),
                     0,
                     cmdLineParams.getAsInt("-iter"),
                     cmdLineParams.getAsFloat("-alpha"),
                     cmdLineParams.getAsInt("-negative_d"),
                     cmdLineParams.getAsInt("-negative_a"),
                     cmdLineParams.getAsInt("-threads") );

    // инициализация нейросети
    trainer.create_net();
    trainer.init_net();

    // запускаем потоки, осуществляющие обучение
    size_t threads_count = cmdLineParams.getAsInt("-threads");
    std::vector<std::thread> threads_vec;
    threads_vec.reserve(threads_count);
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec.emplace_back(&Trainer::train_entry_point, &trainer, i);
    // ждем завершения обучения
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec[i].join();

    // вычисление взвешенного среднего между вектором слова и векторами связанных с ним временных словосочетаний (для которых данное слово является синтакс. вершиной)
    std::vector< std::vector< std::pair<size_t, float> > > collapsing_info;
    v_mwe->process_transient(v_main, collapsing_info);
    trainer.vectors_weighted_collapsing(collapsing_info);
    // TODO: сделать удаление временных словосочетаний из модели

    // сохраняем вычисленные вектора в файл
    if (cmdLineParams.isDefined("-model"))
      trainer.saveEmbeddings( cmdLineParams.getAsString("-model") );
    if (cmdLineParams.isDefined("-backup"))
      trainer.backup( cmdLineParams.getAsString("-backup"), false, true );

    //trainer.print_training_stat();
    return 0;
  } // if task == train

  // если поставлена задача добавления в модель знаков пунктуации
  if (task == "punct")
  {
    AddPunct::run(cmdLineParams.getAsString("-model"));
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

  // если поставлена задача добавления токенов в модель
  if (task == "toks")
  {
    AddToks::run(cmdLineParams.getAsString("-model"), cmdLineParams.getAsString("-tl_map"));
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

    // загрузка векторной модели
    VectorsModel vm;
    if ( !vm.load(cmdLineParams.getAsString("-model")) )
      return -1;

    if ( vm.dep_size == 0 )
    {
      std::cerr << "Assoc. toks isn't trainable." << std::endl;
      return -1;

    }
    if ( !cmdLineParams.isDefined("-restore") )
    {
      std::cerr << "-restore parameter must be defined." << std::endl;
      return -1;
    }

    if ( !cmdLineParams.isDefined("-vocab_d") && vm.dep_size > 0 )
    {
      std::cerr << "-vocab_d parameter must be defined." << std::endl;
      return -1;
    }

    SimpleProfiler global_profiler;

    // загрузка словарей
    std::shared_ptr< OriginalWord2VecVocabulary > v_toks, v_dep_ctx, v_assoc_ctx;
    v_toks = std::make_shared<OriginalWord2VecVocabulary>();
    v_toks->init_whitelist(vm);
    if ( !v_toks->load( cmdLineParams.getAsString("-vocab_t") ) ) // загружаем только те токены, которые уже есть в модели (см. init_whitelist)
      return -1;
    bool needLoadDepCtxVocab = (vm.dep_size > 0);
//    bool needLoadAssocCtxVocab = (vm.assoc_size > 0);
    if (needLoadDepCtxVocab)
    {
      v_dep_ctx = std::make_shared<OriginalWord2VecVocabulary>();
      if ( !v_dep_ctx->load( cmdLineParams.getAsString("-vocab_d") ) )
        return -1;
    }
//    if (needLoadAssocCtxVocab)
//    {
//      v_assoc_ctx = std::make_shared<OriginalWord2VecVocabulary>();
//      v_assoc_ctx->init_stoplist("stopwords.assoc");
//      if ( !v_assoc_ctx->load( cmdLineParams.getAsString("-vocab_t") ) )
//        return -1;
//    }

    // создание поставщика обучающих примеров
    // к моменту создания "поставщика обучающих примеров" словарь должен быть загружен (в частности, используется cn_sum())
    std::shared_ptr< LearningExampleProvider> lep = std::make_shared< LearningExampleProvider > ( cmdLineParams,
                                                                                                  v_toks, true, v_dep_ctx, v_assoc_ctx, nullptr,
                                                                                                  1, false, 0
                                                                                                );
    // создаем объект, организующий обучение
    Trainer trainer( lep, v_toks, true, v_dep_ctx, v_assoc_ctx,
                     vm.dep_size, vm.assoc_size, 0,
                     cmdLineParams.getAsInt("-iter"),
                     cmdLineParams.getAsFloat("-alpha"),
                     cmdLineParams.getAsInt("-negative_d"),
                     cmdLineParams.getAsInt("-negative_a"),
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
    trainer.saveEmbeddings( cmdLineParams.getAsString("-model"), &vm );
    return 0;
  } // if task == toks_train

  // если поставлена задача добавления в модель грамматических эмбеддингов
  if (task == "toks_gramm")
  {
    // загрузим модель
    VectorsModel vm;
    if ( !vm.load(cmdLineParams.getAsString("-model")) )
      return -1;
    // загрузим словарь токенов
    std::shared_ptr< OriginalWord2VecVocabulary > v_toks = std::make_shared<OriginalWord2VecVocabulary>();
    v_toks->init_whitelist(vm);
    if ( !v_toks->load( cmdLineParams.getAsString("-vocab_t") ) )
      return -1;
    // если работаем со словарем суффиксов, то догружаем его в словарь токенов
    std::string oovv = cmdLineParams.getAsString("-vocab_o");
    if (!oovv.empty())
    {
      v_toks->reset_whitelist();
      if ( !v_toks->load(oovv) )
        return -1;
    }
    // создание поставщика обучающих примеров
    std::shared_ptr< LearningExampleProvider> lep = std::make_shared< LearningExampleProvider > ( cmdLineParams,
                                                                                                  v_toks, false, nullptr, nullptr, nullptr,
                                                                                                  1, !oovv.empty(), cmdLineParams.getAsInt("-max_oov_sfx")
                                                                                                );
    // создаем объект, организующий обучение
    Trainer trainer( lep, v_toks, false, nullptr, nullptr,
                     vm.dep_size, vm.assoc_size, cmdLineParams.getAsInt("-size_g"),
                     cmdLineParams.getAsInt("-iter"),
                     cmdLineParams.getAsFloat("-alpha"),
                     cmdLineParams.getAsInt("-negative_d"),
                     cmdLineParams.getAsInt("-negative_a"),
                     cmdLineParams.getAsInt("-threads") );

    // инициализация нейросети
    trainer.create_and_init_gramm_net();

    size_t threads_count = cmdLineParams.getAsInt("-threads");
    // запускаем потоки, осуществляющие обучение
    {
      SimpleProfiler train_profiler;
      std::vector<std::thread> threads_vec;
      threads_vec.reserve(threads_count);
      for (size_t i = 0; i < threads_count; ++i)
        threads_vec.emplace_back(&Trainer::train_entry_point__gramm, &trainer, i);
      // ждем завершения обучения
      for (size_t i = 0; i < threads_count; ++i)
        threads_vec[i].join();
      std::cout << std::endl << "Training finished.";
    }

    // сохраняем вычисленные вектора в файл
    {
      SimpleProfiler saving_profiler;
      trainer.saveGrammaticalEmbeddings( vm, cmdLineParams.getAsFloat("-g_ratio"), oovv, cmdLineParams.getAsString("-model") );
      std::cout << "Embeddings saving finished.";
    }
    return 0;
  } // if task == toks_gramm

  // если поставлена задача балансировки модели (изменения весового соотношения dep и assoc частей)
  if (task == "balance")
  {
    Balancer::run(cmdLineParams.getAsString("-model"), cmdLineParams.getAsFloat("-a_ratio"));
    return 0;
  } // if task == balance

  // если поставлена задача извлечения подмодели (с конвертацией в бинарный word2vec формат)
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
    VectorsModel vm;
    if ( !vm.load(model_fn) )
      return -1;
    FILE *fo = fopen(model_fn.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", vm.words_count, rb-lb);
    for (size_t a = 0; a < vm.vocab.size(); ++a)
      VectorsModel::write_embedding_slice(fo, vm.vocab[a], &vm.embeddings[a * vm.emb_size], lb, rb);
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

  // если поставлена задача оценки вариантов subsampling'а для данного словаря
  if (task == "sseval")
  {
    if ( !cmdLineParams.isDefined("-eval_vocab") )
    {
      std::cerr << "-eval_vocab parameter must be defined." << std::endl;
      return -1;
    }
    SsEval::run( cmdLineParams.getAsString("-eval_vocab") );
    return 0;
  } // if task == sseval

  // если поставлена задача извлечения связных пар из модели
  if (task == "extract")
  {
    RelatedPairsExtractor e;
    e.run(cmdLineParams);
    return 0;
  } // if task == extract

  // если поставлена задача мержинга связных пар, извелченных из нескольких моделей
  if (task == "emerge")
  {
    RelatedPairsExtractor e;
    e.merge(cmdLineParams);
    return 0;
  } // if task == emerge

  // если поставлена задача разделения модели на подмодели псевдооснов, суффиксов и полных слов
  if (task == "spl_m")
  {
    ModelSplitter::run( cmdLineParams.getAsString("-model"), cmdLineParams.getAsString("-tl_map"),
                        cmdLineParams.getAsString("-vocab_o"), cmdLineParams.getAsInt("-size_g") );
    return 0;
  } // if task == spl_m


  // если поставлена задача подготовки эмбеддингов для RUE-модели
  if (task == "rue")
  {
    MakeRueEmbeddings::run(cmdLineParams.getAsString("-model"), cmdLineParams.getAsString("-tl_map"));
    return 0;
  } // if task == toks

  return -1;
}
