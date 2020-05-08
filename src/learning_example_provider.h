#ifndef LEARNING_EXAMPLE_PROVIDER_H_
#define LEARNING_EXAMPLE_PROVIDER_H_

#include "conll_reader.h"

#include <vector>
#include <optional>
#include <cstring>       // for std::strerror


// структура, представляющая обучающий пример
struct LearningExample
{
  size_t word;                         // индекс слова
  std::vector<size_t> dep_context;     // индексы синтаксических контекстов
  std::vector<size_t> assoc_context;   // индексы ассоциативных контекстов
};


// представление токена предложения через индексы слов в разных словарях
struct TokenData
{
  size_t word_idx;
  size_t dep_idx;
  size_t assoc_idx;
  size_t parent_token_no;
  std::vector<size_t> deps;
};

// информация, описывающая рабочий контекст одного потока управления (thread)
struct ThreadEnvironment
{
  FILE* fi;                                            // хэндлер файла, содержащего обучающее множество (открывается с позиции, рассчитанной для данного потока управления).
  std::vector< LearningExample > sentence;             // последнее считанное предложение
  int position_in_sentence;                            // текущая позиция в предложении
  unsigned long long next_random;                      // поле для вычисления случайных величин
  unsigned long long words_count;                      // количество прочитанных словарных слов
  ThreadEnvironment()
  : fi(nullptr)
  , position_in_sentence(-1)
  , next_random(0)
  , words_count(0)
  {
    sentence.reserve(5000);
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
                          size_t embColumn, size_t depColumn)
  : threads_count(threadsCount)
  , train_filename(trainFilename)
  , train_file_size(0)
  , train_words(0)
  , words_vocabulary(wordsVocabulary)
  , dep_ctx_vocabulary(depCtxVocabulary)
  , assoc_ctx_vocabulary(assocCtxVocabulary)
  , emb_column(embColumn)
  , dep_column(depColumn)
  {
    thread_environment.resize(threads_count);
    for (size_t i = 0; i < threads_count; ++i)
      thread_environment[i].next_random = i;
    if ( words_vocabulary )
      train_words = words_vocabulary->cn_sum();
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
        std::vector< std::vector<std::string> > sentence_matrix; // todo: можно вынести в t_environment, а здесь делать только clear (для оптимизации)
        sentence_matrix.reserve(200);
        bool succ = ConllReader::read_sentence(t_environment.fi, sentence_matrix);
        if ( feof(t_environment.fi) ) // не настал ли конец эпохи?
          return std::nullopt;
        if ( !succ )
          continue;
        // проконтролируем, что номер первого токена равен единице
        if (sentence_matrix.size() > 0)
        {
          try {
            int tn = std::stoi( sentence_matrix[0][0] );
            if (tn != 1) continue;
          } catch (...) {
            continue;
          }
        }
        // конвертируем conll-таблицу в более удобную структуру
        const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
        std::vector< TokenData > sentence_idxs;
        for (auto& rec : sentence_matrix)
        {
          TokenData td;
          td.word_idx = words_vocabulary->word_to_idx(rec[emb_column]);
          if ( dep_ctx_vocabulary )
            td.dep_idx = dep_ctx_vocabulary->word_to_idx(rec[dep_column]);
          else
            td.dep_idx = INVALID_IDX;
          if ( assoc_ctx_vocabulary )
            td.assoc_idx = assoc_ctx_vocabulary->word_to_idx(rec[2]);       // lemma column
          else
            td.assoc_idx = INVALID_IDX;
          try {
            td.parent_token_no = std::stoi(rec[6]);  // todo: реализовать быструю конверсию
          } catch (...) {
            td.parent_token_no = 0; // если конвертирование неудачно, считаем, что нет родителя
          }
          sentence_idxs.push_back(td);
        }
        if ( dep_ctx_vocabulary )
        {
          for (size_t i = 0; i < sentence_idxs.size(); ++i)
          {
            if ( sentence_idxs[i].parent_token_no == 0 ) continue;
            if ( sentence_idxs[i].parent_token_no > sentence_idxs.size() ) continue;
            auto& parent = sentence_idxs[ sentence_idxs[i].parent_token_no - 1 ];
            if ( parent.dep_idx != INVALID_IDX )
              sentence_idxs[i].deps.push_back( parent.dep_idx );
            if ( sentence_idxs[i].dep_idx != INVALID_IDX )
              parent.deps.push_back( sentence_idxs[i].dep_idx );
          }
        }
        std::set<size_t> associations;
        if ( assoc_ctx_vocabulary )
        {
          for (auto& rec : sentence_idxs)
          {
            if ( rec.assoc_idx != INVALID_IDX )
              associations.insert(rec.assoc_idx);
          }
        }
        // конвертируем в структуру для итерирования (фильтрация несловарных)
        for (auto& rec : sentence_idxs)
        {
          if ( rec.word_idx != INVALID_IDX )
          {
            LearningExample le;
            le.word = rec.word_idx;
            le.dep_context = rec.deps;
            std::copy(associations.begin(), associations.end(), std::back_inserter(le.assoc_context));   // текущее слово считаем себе ассоциативным
            t_environment.sentence.push_back(le);
          }
        }
        if ( t_environment.sentence.empty() )
          continue;
        t_environment.words_count += t_environment.sentence.size();
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
  size_t threads_count;
  // информация, описывающая рабочие контексты потоков управления (thread)
  std::vector<ThreadEnvironment> thread_environment;
  // имя файла, содержащего обучающее множество (conll)
  std::string train_filename;
  // размер тренировочного файла
  uint64_t train_file_size;
  // количество слов в обучающем множестве (приблизительно, т.к. могло быть подрезание по порогу частоты при построении словаря)
  uint64_t train_words;
  // словари
  std::shared_ptr< OriginalWord2VecVocabulary> words_vocabulary;
  std::shared_ptr< OriginalWord2VecVocabulary> dep_ctx_vocabulary;
  std::shared_ptr< OriginalWord2VecVocabulary> assoc_ctx_vocabulary;
  // номера колонок в conll, откуда считывать данные
  size_t emb_column;
  size_t dep_column;

  // получение размера файла
  uint64_t get_file_size(const std::string& filename)
  {
    // TODO: в будущем использовать std::experimental::filesystem::file_size
    std::ifstream ifs(filename, std::ios::binary|std::ios::ate);
    if ( !ifs.good() )
        throw std::runtime_error(std::strerror(errno));
    return ifs.tellg();
  }
};


#endif /* LEARNING_EXAMPLE_PROVIDER_H_ */
