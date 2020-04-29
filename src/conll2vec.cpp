#include <memory>
#include <string>
#include <thread>

#include "command_line_parameters_defs.h"
#include "simple_profiler.h"
#include "vocabs_builder.h"
#include "original_word2vec_vocabulary.h"
#include "original_word2vec_le_provider.h"
#include "sg_trainer_mikolov.h"



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

  // если поставлена задача построения словарей
  if (task == "vocab")
  {
    VocabsBuilder vb;
    bool succ = vb.build_vocabs( cmdLineParams.getAsString("-train"),
                                 cmdLineParams.getAsString("-vocab_m"), cmdLineParams.getAsString("-vocab_p"),
                                 cmdLineParams.getAsString("-vocab_d"), cmdLineParams.getAsString("-vocab_a"),
                                 cmdLineParams.getAsInt("-min-count_m"), cmdLineParams.getAsInt("-min-count_p"),
                                 cmdLineParams.getAsInt("-min-count_d"), cmdLineParams.getAsInt("-min-count_a"),
                                 cmdLineParams.getAsInt("-col_emb") - 1, cmdLineParams.getAsInt("-col_ctx_d") - 1
                               );
    return ( succ ? 0 : -1 );
  }

  if (!cmdLineParams.isDefined("-words-vocab") || !cmdLineParams.isDefined("-train") || !cmdLineParams.isDefined("-model"))
    return 0;

  SimpleProfiler global_profiler;

  // загрузка словаря
  std::shared_ptr< OriginalWord2VecVocabulary> v = std::make_shared<OriginalWord2VecVocabulary>();
  if ( !v->load( cmdLineParams.getAsString("-words-vocab") ) )
    return -1;

  // создание поставщика обучающих примеров
  // к моменту создания "поставщика обучающих примеров" словарь должен быть загружен (в частности, используется cn_sum())
  std::shared_ptr< CustomLearningExampleProvider> lep = std::make_shared< OriginalWord2VecLearningExampleProvider > ( cmdLineParams.getAsString("-train"),
                                                                                                                      cmdLineParams.getAsInt("-threads"),
                                                                                                                      cmdLineParams.getAsInt("-window"),
                                                                                                                      cmdLineParams.getAsFloat("-sample"),
                                                                                                                      v );

  // создаем объект, организующий обучение
  SgTrainer_Mikolov trainer( lep, v , v,
                       cmdLineParams.getAsInt("-size"),
                       cmdLineParams.getAsInt("-iter"),
                       cmdLineParams.getAsFloat("-alpha"),
                       cmdLineParams.getAsFloat("-negative"));

  // инициализация нейросети
  trainer.init_net();

  // запускаем потоки, осуществляющие обучение
  size_t threads_count = cmdLineParams.getAsInt("-threads");
  std::vector<std::thread> threads_vec;
  threads_vec.reserve(threads_count);
  for (size_t i = 0; i < threads_count; ++i)
    threads_vec.emplace_back(&CustomTrainer::train_entry_point, &trainer, i);
  for (size_t i = 0; i < threads_count; ++i)
    threads_vec[i].join();

  // сохраняем вычисленные вектора в файл
  trainer.saveEmbeddings( cmdLineParams.getAsString("-model") );
//  if (cmdLineParams.isDefined("-backup"))
//    trainer.backup( cmdLineParams.getAsString("-backup") );

  return 0;
}
