#include "command_line_parameters_defs.h"
#include "simple_profiler.h"
#include "fit_parus.h"
#include "vocabs_builder.h"
#include "original_word2vec_vocabulary.h"
#include "learning_example_provider.h"
#include "trainer.h"
#include "sim_estimator.h"

#include <memory>
#include <string>
#include <iostream>
#include <thread>



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
              << "  -task fit   -- conll file transformation" << std::endl
              << "  -task vocab -- vocabs building" << std::endl
              << "  -task train -- model training" << std::endl
              << "  -task punct -- add punctuation to model " << std::endl
              << "  -task sim   -- similarity test" << std::endl;
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
                                 cmdLineParams.getAsString("-vocab_m"), cmdLineParams.getAsString("-vocab_p"),
                                 cmdLineParams.getAsString("-vocab_d"), cmdLineParams.getAsString("-vocab_a"),
                                 cmdLineParams.getAsInt("-min-count_m"), cmdLineParams.getAsInt("-min-count_p"),
                                 cmdLineParams.getAsInt("-min-count_d"), cmdLineParams.getAsInt("-min-count_a"),
                                 cmdLineParams.getAsInt("-col_emb") - 1, cmdLineParams.getAsInt("-col_ctx_d") - 1,
                                 (cmdLineParams.getAsInt("-use_deprel") == 1)
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
      if ( !v_assoc_ctx->load( cmdLineParams.getAsString("-vocab_a") ) )
        return -1;
    }

    // создание поставщика обучающих примеров
    // к моменту создания "поставщика обучающих примеров" словарь должен быть загружен (в частности, используется cn_sum())
    std::shared_ptr< LearningExampleProvider> lep = std::make_shared< LearningExampleProvider > ( cmdLineParams.getAsString("-train"),
                                                                                                  cmdLineParams.getAsInt("-threads"),
                                                                                                  (needLoadMainVocab ? v_main : v_proper ),
                                                                                                  v_dep_ctx, v_assoc_ctx,
                                                                                                  cmdLineParams.getAsInt("-col_emb") - 1,
                                                                                                  cmdLineParams.getAsInt("-col_ctx_d") - 1,
                                                                                                  (cmdLineParams.getAsInt("-use_deprel") == 1),
                                                                                                  cmdLineParams.getAsFloat("-sample_a")
                                                                                                );

    // создаем объект, организующий обучение
    Trainer trainer( lep, (needLoadMainVocab ? v_main : v_proper ), needLoadProperVocab,
                     v_dep_ctx, v_assoc_ctx,
                     cmdLineParams.getAsInt("-size_d"),
                     cmdLineParams.getAsInt("-size_a"),
                     cmdLineParams.getAsInt("-iter"),
                     cmdLineParams.getAsFloat("-alpha"),
                     cmdLineParams.getAsInt("-negative"),
//                     cmdLineParams.getAsFloat("-zerolize"),
//                     cmdLineParams.getAsFloat("-space_lim"),
                     cmdLineParams.getAsFloat("-z_reg_d"),
                     cmdLineParams.getAsFloat("-z_reg_a"),
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
      trainer.restore( cmdLineParams.getAsString("-restore"), false, true );
    }

    // запускаем поток, обеспечивающий вычисление параметров регуляризации
    std::thread regularization_thread(&Trainer::regularization_entry_point, &trainer);
    // запускаем потоки, осуществляющие обучение
    size_t threads_count = cmdLineParams.getAsInt("-threads");
    std::vector<std::thread> threads_vec;
    threads_vec.reserve(threads_count);
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec.emplace_back(&Trainer::train_entry_point, &trainer, i);
    // ждем завершения обучения
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec[i].join();
    trainer.all_done_signal();
    regularization_thread.join();

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

  // если поставлена задача оценки близости значений (в интерактивном режиме)
  if (task == "sim")
  {
    SimilarityEstimator sim_estimator(cmdLineParams.getAsInt("-size_d"), cmdLineParams.getAsInt("-size_a"));
    if ( !sim_estimator.load_model(cmdLineParams.getAsString("-model"), (cmdLineParams.getAsString("-model_fmt") == "txt")) )
      return -1;
    sim_estimator.run();
    return 0;
  } // if task == sim

  return -1;
}
