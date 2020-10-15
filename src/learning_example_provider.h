#ifndef LEARNING_EXAMPLE_PROVIDER_H_
#define LEARNING_EXAMPLE_PROVIDER_H_

#include "conll_reader.h"
#include "original_word2vec_vocabulary.h"

#include <memory>
#include <vector>
#include <optional>
#include <cstring>       // for std::strerror
#include <cmath>


// структура, представляющая обучающий пример
struct LearningExample
{
  size_t word;                         // индекс слова
  std::vector<size_t> dep_context;     // индексы синтаксических контекстов
  std::vector<size_t> assoc_context;   // индексы ассоциативных контекстов
};


// информация, описывающая рабочий контекст одного потока управления (thread)
struct ThreadEnvironment
{
  FILE* fi;                                            // хэндлер файла, содержащего обучающее множество (открывается с позиции, рассчитанной для данного потока управления).
  std::vector< LearningExample > sentence;             // последнее считанное предложение
  int position_in_sentence;                            // текущая позиция в предложении
  unsigned long long next_random;                      // поле для вычисления случайных величин
  unsigned long long words_count;                      // количество прочитанных словарных слов
  std::vector< std::vector<std::string> > sentence_matrix; // conll-матрица для предложения
  ThreadEnvironment()
  : fi(nullptr)
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



// Класс поставщика обучающих примеров ("итератор" по обучающему множеству).
// Выдает обучающие примеры в терминах индексов в словарях (полностью закрывает собой слова-строки).
class LearningExampleProvider
{
public:
  // конструктор
  LearningExampleProvider(const std::string& trainFilename, size_t threadsCount,
                          std::shared_ptr< OriginalWord2VecVocabulary> wordsVocabulary,
                          std::shared_ptr< OriginalWord2VecVocabulary> depCtxVocabulary, std::shared_ptr< OriginalWord2VecVocabulary> assocCtxVocabulary,
                          size_t embColumn, size_t depColumn, bool useDeprel,
                          float wordsSubsample, float depSubsample, float assocSubsample)
  : threads_count(threadsCount)
  , train_filename(trainFilename)
  , words_vocabulary(wordsVocabulary)
  , dep_ctx_vocabulary(depCtxVocabulary)
  , assoc_ctx_vocabulary(assocCtxVocabulary)
  , emb_column(embColumn)
  , dep_column(depColumn)
  , use_deprel(useDeprel)
  , sample_w(wordsSubsample)
  , sample_d(depSubsample)
  , sample_a(assocSubsample)
  {
    thread_environment.resize(threads_count);
    for (size_t i = 0; i < threads_count; ++i)
      thread_environment[i].next_random = i;
    if ( words_vocabulary )
      train_words = words_vocabulary->cn_sum();
    w_mul_sample_w = train_words * sample_w;
    if ( dep_ctx_vocabulary )
      train_words_dep = dep_ctx_vocabulary->cn_sum();
    w_mul_sample_d = train_words_dep * sample_d;
    if ( assoc_ctx_vocabulary )
      train_words_assoc = assoc_ctx_vocabulary->cn_sum();
    w_mul_sample_a = train_words_assoc * sample_a;
    try
    {
      train_file_size = get_file_size(train_filename);
    } catch (const std::runtime_error& e) {
      std::cerr << "LearningExampleProvider can't get file size for: " << train_filename << "\n  " << e.what() << std::endl;
      train_file_size = 0;
    }
  } // constructor-end
  // деструктор
  ~LearningExampleProvider()
  {
  }
  // подготовительные действия, выполняемые перед каждой эпохой обучения
  bool epoch_prepare(size_t threadIndex)
  {
    auto& t_environment = thread_environment[threadIndex];
    if (train_file_size == 0)
      return false;
    t_environment.fi = fopen(train_filename.c_str(), "rb");
    if ( t_environment.fi == nullptr )
    {
      std::cerr << "LearningExampleProvider: epoch prepare error: " << std::strerror(errno) << std::endl;
      return false;
    }
    int succ = fseek(t_environment.fi, train_file_size / threads_count * threadIndex, SEEK_SET);
    if (succ != 0)
    {
      std::cerr << "LearningExampleProvider: epoch prepare error: " << std::strerror(errno) << std::endl;
      return false;
    }
    // т.к. после смещения мы типично не оказываемся в начале предложения, выполним выравнивание на начало предложения
    std::vector< std::vector<std::string> > stub;
    ConllReader::read_sentence(t_environment.fi, stub); // один read_sentence не гарантирует выход на начало предложения, т.к. fseek может поставить нас прямо на перевод строки в конце очередного токена, что распознается, как пустая строка
    stub.clear();
    ConllReader::read_sentence(t_environment.fi, stub);
    t_environment.sentence.clear();
    t_environment.position_in_sentence = 0;
    t_environment.words_count = 0;
    return true;
  } // method-end
  // заключительные действия, выполняемые после каждой эпохой обучения
  bool epoch_unprepare(size_t threadIndex)
  {
    auto& t_environment = thread_environment[threadIndex];
    fclose( t_environment.fi );
    t_environment.fi = nullptr;
    return true;
  } // method-end
  // получение очередного обучающего примера
  std::optional<LearningExample> get(size_t threadIndex)
  {
    auto& t_environment = thread_environment[threadIndex];

    if (t_environment.sentence.empty())
    {
      t_environment.position_in_sentence = 0;
      if ( t_environment.words_count > train_words / threads_count ) // не настал ли конец эпохи?
        return std::nullopt;
      while (true)
      {
        auto& sentence_matrix = t_environment.sentence_matrix;
        sentence_matrix.clear();
        bool succ = ConllReader::read_sentence(t_environment.fi, sentence_matrix);
        if ( feof(t_environment.fi) ) // не настал ли конец эпохи?
          return std::nullopt;
        if ( !succ )
          continue;
        auto sm_size = sentence_matrix.size();
        if (sm_size == 0)
          continue;
        // проконтролируем, что номер первого токена равен единице
        try {
          int tn = std::stoi( sentence_matrix[0][0] );
          if (tn != 1) continue;
        } catch (...) {
          continue;
        }
        // конвертируем conll-таблицу в более удобные структуры
        const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
        std::vector< std::vector<size_t> > deps( sm_size );  // хранилище синатксических контекстов для каждого токена
        std::set<size_t> associations; // хранилище ассоциативных контекстов для всего предложения
        if ( dep_ctx_vocabulary )
        {
          for (size_t i = 0; i < sm_size; ++i)
          {
            auto& token = sentence_matrix[i];
            size_t parent_token_no = 0;
            try {
              parent_token_no = std::stoi(token[6]);
            } catch (...) {
              parent_token_no = 0; // если конвертирование неудачно, считаем, что нет родителя
            }
            if ( parent_token_no < 1 || parent_token_no > sm_size ) continue;

            // рассматриваем контекст с точки зрения родителя в синтаксической связи
            auto ctx__from_head_viewpoint = ( use_deprel ? token[dep_column] + "<" + token[7] : token[dep_column] );
            auto ctx__fhvp_idx = dep_ctx_vocabulary->word_to_idx( ctx__from_head_viewpoint );
            if ( ctx__fhvp_idx != INVALID_IDX )
              deps[ parent_token_no - 1 ].push_back( ctx__fhvp_idx );
            // рассматриваем контекст с точки зрения потомка в синтаксической связи
            auto& parent = sentence_matrix[ parent_token_no - 1 ];
            auto ctx__from_child_viewpoint = (use_deprel ? parent[dep_column] + ">" + token[7] : parent[dep_column] );
            auto ctx__fcvp_idx = dep_ctx_vocabulary->word_to_idx( ctx__from_child_viewpoint );
            if ( ctx__fcvp_idx != INVALID_IDX )
              deps[ i ].push_back( ctx__fcvp_idx );
          }
          if (sample_d > 0)
          {
            for (size_t i = 0; i < sm_size; ++i)
            {
              auto tdcIt = deps[i].begin();
              while (tdcIt != deps[i].end())
              {
                auto&& dep_record = dep_ctx_vocabulary->idx_to_data(*tdcIt);
                float ran = (std::sqrt(dep_record.cn / (w_mul_sample_d)) + 1) * (w_mul_sample_d) / dep_record.cn;
                t_environment.update_random();
                if (ran < (t_environment.next_random & 0xFFFF) / (float)65536)
                  tdcIt = deps[i].erase(tdcIt);
                else
                  ++tdcIt;
              }
            }
          }
        }
        if ( assoc_ctx_vocabulary )
        {
          for (auto& rec : sentence_matrix)
          {
            size_t assoc_idx = assoc_ctx_vocabulary->word_to_idx(rec[2]);       // lemma column
            if ( assoc_idx != INVALID_IDX )
            {
              // применяем сабсэмплинг к ассоциациям
              if (sample_a > 0)
              {
                  auto&& assoc_record = assoc_ctx_vocabulary->idx_to_data(assoc_idx);
                  float ran = (std::sqrt(assoc_record.cn / (w_mul_sample_a)) + 1) * (w_mul_sample_a) / assoc_record.cn;
                  t_environment.update_random();
                  if (ran < (t_environment.next_random & 0xFFFF) / (float)65536)
                    continue;
              }
              associations.insert(assoc_idx);
            }
          }
        }
        // конвертируем в структуру для итерирования (фильтрация несловарных)
        for (size_t i = 0; i < sm_size; ++i)
        {
          auto word_idx = words_vocabulary->word_to_idx(sentence_matrix[i][emb_column]);
          if ( word_idx != INVALID_IDX )
          {
            ++t_environment.words_count;
            if (sample_w > 0)
            {
              auto&& w_record = words_vocabulary->idx_to_data(word_idx);
              float ran = (std::sqrt(w_record.cn / (w_mul_sample_w)) + 1) * (w_mul_sample_w) / w_record.cn;
              t_environment.update_random();
              if (ran < (t_environment.next_random & 0xFFFF) / (float)65536)
                continue;
            }
            LearningExample le;
            le.word = word_idx;
            le.dep_context = deps[i];
            //std::copy(associations.begin(), associations.end(), std::back_inserter(le.assoc_context));   // текущее слово считаем себе ассоциативным
            std::copy_if( associations.begin(), associations.end(), std::back_inserter(le.assoc_context),
                          [word_idx](const size_t a_idx) {return (a_idx != word_idx);} );                  // текущее слово не считаем себе ассоциативным
            t_environment.sentence.push_back(le);
          }
        }
        if ( t_environment.sentence.empty() )
          continue;
        break;
      }
    }

    // при выходе из цикла выше в t_environment.sentence должно быть полезное предложение
    // итерируем по нему
    LearningExample result;
    result = t_environment.sentence[t_environment.position_in_sentence++];
    if ( t_environment.position_in_sentence == static_cast<int>(t_environment.sentence.size()) )
      t_environment.sentence.clear();
    return result;
  } // method-end
  // получение количества слов, фактически считанных из обучающего множества (т.е. без учета сабсэмплинга)
  uint64_t getWordsCount(size_t threadIndex) const
  {
    return thread_environment[threadIndex].words_count;
  }
private:
  // количество потоков управления (thread), параллельно работающих с поставщиком обучающих примеров
  size_t threads_count = 0;
  // информация, описывающая рабочие контексты потоков управления (thread)
  std::vector<ThreadEnvironment> thread_environment;
  // имя файла, содержащего обучающее множество (conll)
  std::string train_filename;
  // размер тренировочного файла
  uint64_t train_file_size = 0;
  // количество слов в обучающем множестве (приблизительно, т.к. могло быть подрезание по порогу частоты при построении словаря)
  uint64_t train_words = 0;
  // количество слов в обучающем множестве, вошедших в словарь синтаксических контекстов (приблизительно, т.к. могло быть подрезание по порогу частоты при построении словаря)
  uint64_t train_words_dep = 0;
  // количество слов в обучающем множестве, вошедших в словарь ассоциативных контекстов (приблизительно, т.к. могло быть подрезание по порогу частоты при построении словаря)
  uint64_t train_words_assoc = 0;
  // словари
  std::shared_ptr< OriginalWord2VecVocabulary > words_vocabulary;
  std::shared_ptr< OriginalWord2VecVocabulary > dep_ctx_vocabulary;
  std::shared_ptr< OriginalWord2VecVocabulary > assoc_ctx_vocabulary;
  // номера колонок в conll, откуда считывать данные
  size_t emb_column;
  size_t dep_column;
  // следует ли задействовать тип и направление синтаксической связи в определении синтаксического контекста
  bool use_deprel;
  // порог для алгоритма сэмплирования (subsampling) -- для словаря векторной модели
  float sample_w = 1e-3;
  // порог для алгоритма сэмплирования (subsampling) -- для синтаксических контекстов
  float sample_d = 1e-3;
  // порог для алгоритма сэмплирования (subsampling) -- для ассоциативных контекстов
  float sample_a = 1e-3;
  // хранилища произведений порога сэмплирования и количества слов в соответствующем словаре (вычислительная оптимизация)
  float w_mul_sample_w = 0, w_mul_sample_d = 0, w_mul_sample_a = 0;

  // получение размера файла
  uint64_t get_file_size(const std::string& filename)
  {
    // TODO: в будущем использовать std::experimental::filesystem::file_size
    std::ifstream ifs(filename, std::ios::binary|std::ios::ate);
    if ( !ifs.good() )
        throw std::runtime_error(std::strerror(errno));
    return ifs.tellg();
  } // method-end
//  // быстрый конвертер строки в число (без какого-либо контроля корректности)
//  unsigned int string2uint_ultrafast(const std::string& value)
//  {
//    const char* str = value.c_str();
//    unsigned int val = 0;
//    while( *str )
//      val = val*10 + (*str++ - '0');
//    return val;
//  } // method-end
};


#endif /* LEARNING_EXAMPLE_PROVIDER_H_ */
