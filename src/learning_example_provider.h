#ifndef LEARNING_EXAMPLE_PROVIDER_H_
#define LEARNING_EXAMPLE_PROVIDER_H_

#include "conll_reader.h"
#include "original_word2vec_vocabulary.h"
#include "str_conv.h"
#include "learning_example.h"
#include "command_line_parameters_defs.h"

#include <memory>
#include <vector>
#include <optional>
#include <cstring>       // for std::strerror

//#include "log.h"

// информация, описывающая рабочий контекст одного потока управления (thread)
struct ThreadEnvironment
{
  std::unique_ptr<ConllReader> cr;
  std::vector< LearningExample > sentence;             // последнее считанное предложение
  int position_in_sentence;                            // текущая позиция в предложении
  unsigned long long next_random;                      // поле для вычисления случайных величин
  unsigned long long words_count;                      // количество прочитанных словарных слов
  ConllReader::SentenceMatrix sentence_matrix;         // conll-матрица для предложения
  ThreadEnvironment()
  : cr(nullptr)
  , position_in_sentence(-1)
  , next_random(0)
  , words_count(0)
  {
    sentence.reserve(1000);
    sentence_matrix.reserve(1000);
  }
  inline void update_random()
  {
    next_random = next_random * (unsigned long long)25214903917 + 11;
  }
};



// Базовый класс поставщика обучающих примеров ("итератор" по обучающему множеству).
// Выдает обучающие примеры в терминах индексов в словарях (полностью закрывает собой слова-строки).
class LearningExampleProvider
{
public:

  // конструктор
  LearningExampleProvider( const CommandLineParametersDefs& cmdLineParams,
                           std::shared_ptr<OriginalWord2VecVocabulary> wordsVocabulary )
  : threads_count( cmdLineParams.getAsInt("-threads") )
  , train_filename( cmdLineParams.getAsString("-train") )
  , words_vocabulary(wordsVocabulary)
  , sample_w( cmdLineParams.getAsFloat("-sample_w") )
  {
    thread_environment.resize(threads_count);
    for (size_t i = 0; i < threads_count; ++i)
      thread_environment[i].next_random = i;
    if ( words_vocabulary )
    {
      train_words = words_vocabulary->cn_sum();
      words_vocabulary->sampling_estimation(sample_w);
    }
    for (size_t i = 0; i < threads_count; ++i)
      thread_environment[i].cr = std::make_unique<ConllReader>(train_filename);
  } // constructor-end

  // подготовительные действия, выполняемые перед каждой эпохой обучения
  bool epoch_prepare(size_t threadIndex)
  {
    auto& t_environment = thread_environment[threadIndex];
    if ( !t_environment.cr->init_multithread(threadIndex, threads_count) )
    {
      std::cerr << "LearningExampleProvider: epoch prepare error" << std::endl;
      return false;
    }
    t_environment.sentence.clear();
    t_environment.position_in_sentence = 0;
    t_environment.words_count = 0;
    return true;
  } // method-end

  // заключительные действия, выполняемые после каждой эпохой обучения
  bool epoch_unprepare(size_t threadIndex)
  {
    auto& t_environment = thread_environment[threadIndex];
    t_environment.cr->fin();
    return true;
  } // method-end

  // получение очередного обучающего примера
  // todo: в c++20 перейти на coroutines?
  std::optional<LearningExample> get(size_t threadIndex, float fraction, bool gramm = false)
  {
    auto& t_environment = thread_environment[threadIndex];

    if (t_environment.sentence.empty())
    {
      t_environment.position_in_sentence = 0;
      if ( t_environment.words_count > train_words / threads_count ) // не настал ли конец эпохи?
        return std::nullopt;
      auto& sentence_matrix = t_environment.sentence_matrix;
      do
      {

        bool is_read_ok = t_environment.cr->read_sentence(sentence_matrix);
        if ( !is_read_ok ) // не настал ли конец эпохи? (вычитали весь файл или ошибка чтения)
          return std::nullopt;
        if ( sentence_matrix.empty() ) // предохранитель
          return std::nullopt;

        get_lp_specific(t_environment, fraction);

      } while ( t_environment.sentence.empty() );
    }

    // при выходе из цикла выше в t_environment.sentence должно быть полезное предложение
    // итерируем по нему
    LearningExample result;
    result = t_environment.sentence[t_environment.position_in_sentence++];
    if ( t_environment.position_in_sentence == static_cast<int>(t_environment.sentence.size()) )
      t_environment.sentence.clear();
    return result;
  } // method-end

  // настоящий get, переопределяемый в классах потомках
  virtual void get_lp_specific(ThreadEnvironment& t_environment, float fraction) = 0;

  // получение количества слов, фактически считанных из обучающего множества (т.е. без учета сабсэмплинга)
  uint64_t getWordsCount(size_t threadIndex) const
  {
    return thread_environment[threadIndex].words_count;
  }

  // изменение subsampling-коэффициентов в динамике
  void update_subsampling_rates(float w_mul = 0.7 /*, float d_mul = 0.95, float a_mul = 0.95*/)
  {
    sample_w *= w_mul; /*sample_d *= d_mul; sample_a *= a_mul;*/
    if ( words_vocabulary )
      words_vocabulary->sampling_estimation(sample_w);
    // if ( dep_ctx_vocabulary )
    //   dep_ctx_vocabulary->sampling_estimation(sample_d);
    // if ( assoc_ctx_vocabulary )
    //   assoc_ctx_vocabulary->sampling_estimation(sample_a);
  }

protected:
  // основной словарь
  std::shared_ptr< OriginalWord2VecVocabulary > words_vocabulary;
  // порог для алгоритма сэмплирования (subsampling) -- для словаря векторной модели
  float sample_w = 0;

  // для каждого слова в sentence_matrix получение индексов (в той же sentence_matrix) синтаксически связанных слов
  static std::vector<std::set<size_t>> get_syntactically_related(const ConllReader::SentenceMatrix& sentence_matrix)
  {
    auto sm_size = sentence_matrix.size();
    std::vector<std::set<size_t>> result( sm_size );
    for (size_t i = 0; i < sm_size; ++i)
    {
      auto& token = sentence_matrix[i];
      if ( token[Conll::DEPREL] == "_" ) continue;
      size_t parent_token_no = 0;
      try {
        parent_token_no = std::stoi(token[Conll::HEAD]);
      } catch (...) {
        parent_token_no = 0; // если конвертирование неудачно, считаем, что нет родителя
      }
      if ( parent_token_no < 1 || parent_token_no > sm_size ) continue;
      result[ parent_token_no - 1 ].insert(i);
      result[ i ].insert(parent_token_no - 1);
    }
    return result;
  }

//  // быстрый конвертер строки в число (без какого-либо контроля корректности)
//  unsigned int string2uint_ultrafast(const std::string& value)
//  {
//    const char* str = value.c_str();
//    unsigned int val = 0;
//    while( *str )
//      val = val*10 + (*str++ - '0');
//    return val;
//  } // method-end

private:
  // количество потоков управления (thread), параллельно работающих с поставщиком обучающих примеров
  size_t threads_count = 0;
  // информация, описывающая рабочие контексты потоков управления (thread)
  std::vector<ThreadEnvironment> thread_environment;
  // имя файла, содержащего обучающее множество (conll)
  std::string train_filename;
  // количество слов в обучающем множестве (приблизительно, т.к. могло быть подрезание по порогу частоты при построении словаря)
  uint64_t train_words = 0;

}; // class-decl-end


#endif /* LEARNING_EXAMPLE_PROVIDER_H_ */
