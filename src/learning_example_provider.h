#ifndef LEARNING_EXAMPLE_PROVIDER_H_
#define LEARNING_EXAMPLE_PROVIDER_H_

#include "conll_reader.h"
#include "original_word2vec_vocabulary.h"
#include "str_conv.h"

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
                          std::shared_ptr<OriginalWord2VecVocabulary> wordsVocabulary,
                          bool trainProperNames,
                          std::shared_ptr<OriginalWord2VecVocabulary> depCtxVocabulary, std::shared_ptr<OriginalWord2VecVocabulary> assocCtxVocabulary,
                          size_t embColumn, size_t depColumn, bool useDeprel, bool oov, size_t oovMaxLen,
                          float wordsSubsample, float depSubsample, float assocSubsample)
  : threads_count(threadsCount)
  , train_filename(trainFilename)
  , words_vocabulary(wordsVocabulary)
  , proper_names(trainProperNames)
  , dep_ctx_vocabulary(depCtxVocabulary)
  , assoc_ctx_vocabulary(assocCtxVocabulary)
  , emb_column(embColumn)
  , dep_column(depColumn)
  , use_deprel(useDeprel)
  , train_oov(oov)
  , max_oov_sfx(oovMaxLen)
  , sample_w(wordsSubsample)
  , sample_d(depSubsample)
  , sample_a(assocSubsample)
  {
    thread_environment.resize(threads_count);
    for (size_t i = 0; i < threads_count; ++i)
      thread_environment[i].next_random = i;
    if ( words_vocabulary )
    {
      train_words = words_vocabulary->cn_sum();
      words_vocabulary->sampling_estimation(wordsSubsample);
    }
    if ( dep_ctx_vocabulary )
      dep_ctx_vocabulary->sampling_estimation(depSubsample);
    if ( assoc_ctx_vocabulary )
      assoc_ctx_vocabulary->sampling_estimation(assocSubsample);
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
  std::optional<LearningExample> get(size_t threadIndex, bool gramm = false)
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

        if (!gramm)
          get_from_sentence__usual(t_environment);
        else
          get_from_sentence__gram(t_environment);

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
  // извлечение обучающих примеров из предложения (вспомогат. процедура для get)
  void get_from_sentence__usual(ThreadEnvironment& t_environment)
  {
    auto& sentence_matrix = t_environment.sentence_matrix;
    auto sm_size = sentence_matrix.size();
    const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
    // конвертируем conll-таблицу в более удобные структуры
    std::vector< std::vector<size_t> > deps( sm_size );  // хранилище синатксических контекстов для каждого токена
    std::set<size_t> associations;                       // хранилище ассоциативных контекстов для всего предложения
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
            float ran = dep_ctx_vocabulary->idx_to_data(*tdcIt).sample_probability;
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
        if ( assoc_idx == INVALID_IDX )
          continue;
        // применяем сабсэмплинг к ассоциациям
        if (sample_a > 0)
        {
          float ran = assoc_ctx_vocabulary->idx_to_data(assoc_idx).sample_probability;
          t_environment.update_random();
          if (ran < (t_environment.next_random & 0xFFFF) / (float)65536)
            continue;
        }
        if (!proper_names)
        {
          auto word_idx = words_vocabulary->word_to_idx(rec[emb_column]);
          if ( word_idx == INVALID_IDX )
            continue;
          associations.insert(word_idx);
        }
        else
          associations.insert(assoc_idx);
      } // for all words in sentence
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
          float ran = words_vocabulary->idx_to_data(word_idx).sample_probability;
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
  } // method-end
  // извлечение из предложения обучающих примеров для построения грамматических векторов (вспомогат. процедура для get)
  void get_from_sentence__gram(ThreadEnvironment& t_environment)
  {
    auto& sentence_matrix = t_environment.sentence_matrix;
    auto sm_size = sentence_matrix.size();
    const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < sm_size; ++i)
    {
      auto token_str = sentence_matrix[i][emb_column];            // TODO: возможно по словарю модели, но там не для всех слов известны частоты
      auto word_idx = words_vocabulary->word_to_idx(token_str);
      if ( word_idx != INVALID_IDX )
      {
        ++t_environment.words_count;
        if (sample_w > 0)
        {
          float ran = words_vocabulary->idx_to_data(word_idx).sample_probability;
          t_environment.update_random();
          if (ran < (t_environment.next_random & 0xFFFF) / (float)65536)
            continue;
        }
        LearningExample le;
        le.word = word_idx;
        msd2vec(le, sentence_matrix[i][5]);  // конструирование грамматического вектора из набора граммем
        t_environment.sentence.push_back(le);
      }
      if (train_oov)
        try_to_get_oov_suffixes(t_environment, token_str, sentence_matrix[i][5]);
    }
  } // method-end
  void try_to_get_oov_suffixes(ThreadEnvironment& t_environment, const std::string& token, const std::string& msd)
  {
    const std::string OOV = "_OOV_";
    const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
    auto t32 = StrConv::To_UTF32(token);
    auto wl = t32.length();
    std::string sfx;
    for (size_t i = 0; i < max_oov_sfx; ++i)
    {
      if (wl <= i)
        break;
      sfx = StrConv::To_UTF8(std::u32string(1, t32[wl-i-1])) + sfx;
      auto oov_str = OOV+sfx;
      auto word_idx = words_vocabulary->word_to_idx(oov_str);
      if ( word_idx != INVALID_IDX )
      {
        ++t_environment.words_count;
        if (sample_w > 0)
        {
          float ran = words_vocabulary->idx_to_data(word_idx).sample_probability;
          t_environment.update_random();
          if (ran < (t_environment.next_random & 0xFFFF) / (float)65536)
            continue; // попробуем другие суффиксы, этот пропустим
        }
        LearningExample le;
        le.word = word_idx;
        msd2vec(le, msd);  // конструирование грамматического вектора из набора граммем
        t_environment.sentence.push_back(le);
      }
    }
  } // method-end
  // получение количества слов, фактически считанных из обучающего множества (т.е. без учета сабсэмплинга)
  uint64_t getWordsCount(size_t threadIndex) const
  {
    return thread_environment[threadIndex].words_count;
  }
  // получение длины вектора граммем
  size_t getGrammemesVectorSize() const
  {
    return gcLast;
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
  // словари
  std::shared_ptr< OriginalWord2VecVocabulary > words_vocabulary;
  bool proper_names;  // признак того, что выполняется обучение векторных представлений для собственных имен
  std::shared_ptr< OriginalWord2VecVocabulary > dep_ctx_vocabulary;
  std::shared_ptr< OriginalWord2VecVocabulary > assoc_ctx_vocabulary;
  // номера колонок в conll, откуда считывать данные
  size_t emb_column;
  size_t dep_column;
  // следует ли задействовать тип и направление синтаксической связи в определении синтаксического контекста
  bool use_deprel;
  // нужно ли обучать oov-суффиксы
  bool train_oov;
  // максимальная длина oov-суффикса
  size_t max_oov_sfx;
  // порог для алгоритма сэмплирования (subsampling) -- для словаря векторной модели
  float sample_w = 0;
  // порог для алгоритма сэмплирования (subsampling) -- для синтаксических контекстов
  float sample_d = 0;
  // порог для алгоритма сэмплирования (subsampling) -- для ассоциативных контекстов
  float sample_a = 0;

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

private:
  enum GramCode2VecPosition
  {
    gcPosNoun = 0,
    gcPosPron,
    gcPosAdj,
    gcPosNumeral,
    gcPosAdv,
    gcPosVerb,
    gcPosAdpos,
    gcPosConj,
    gcPosPart,
    gcPosInter,
    gcNtCmn,
    gcNtProper,
    gcGendMas,
    gcGendFem,
    gcGendNeu,
    gcNumSing,
    gcNumPlur,
    gcCaseNom,
    gcCaseGen,
    gcCaseDat,
    gcCaseAcc,
    gcCaseIns,
    gcCaseLoc,
    gcCaseVoc,
    gcAnimYes,
    gcAnimNo,
    gcDegrPos,
    gcDegrCom,
    gcDegrSup,
    gcDefShort,
    gcDefFull,
    gcPossess,
    gcVfInd,
    gcVfImp,
    gcVfCond,
    gcVfInf,
    gcVfPart,
    gcVfGer,
    gcTensePre,
    gcTensePast,
    gcTenseFut,
    gcPers1,
    gcPers2,
    gcPers3,
    gcVoiceAct,
    gcVoicePass,
    gcAspProg,
    gcAspPerf,
    gcPrntPers,
    gcPrntDem,
    gcPrntIndef,
    gcPrntInterrog,
    gcPrntRelat,
    gcPrntReflex,
    gcPrntNeg,
    gcPrntNspec,
    gcStNom,
    gcStAdj,
    gcStAdv,
    gcNumeralCard,
    gcNumeralOrd,
    gcNumeralCollect,

    gcLast
  };
  void encodeGender(char value, LearningExample& le)
  {
    switch(value)
    {
    case 'm': le.assoc_context[gcGendMas] = 1; break;
    case 'f': le.assoc_context[gcGendFem] = 1; break;
    case 'n': le.assoc_context[gcGendNeu] = 1; break;
    }
  }
  void encodeNumber(char value, LearningExample& le)
  {
    switch(value)
    {
    case 's': le.assoc_context[gcNumSing] = 1; break;
    case 'p': le.assoc_context[gcNumPlur] = 1; break;
    }
  }
  void encodeCase(char value, LearningExample& le)
  {
    switch (value)
    {
    case 'n': le.assoc_context[gcCaseNom] = 1; break;
    case 'g': le.assoc_context[gcCaseGen] = 1; break;
    case 'd': le.assoc_context[gcCaseDat] = 1; break;
    case 'a': le.assoc_context[gcCaseAcc] = 1; break;
    case 'i': le.assoc_context[gcCaseIns] = 1; break;
    case 'l': le.assoc_context[gcCaseLoc] = 1; break;
    case 'v': le.assoc_context[gcCaseVoc] = 1; break;
    }
  }
  void encodeAnim(char value, LearningExample& le)
  {
    switch(value)
    {
    case 'y': le.assoc_context[gcAnimYes] = 1; break;
    case 'n': le.assoc_context[gcAnimNo] = 1; break;
    }
  }
  void encodeTense(char value, LearningExample& le)
  {
    switch(value)
    {
    case 'p': le.assoc_context[gcTensePre] = 1; break;
    case 'f': le.assoc_context[gcTenseFut] = 1; break;
    case 's': le.assoc_context[gcTensePast] = 1; break;
    }
  }
  void encodePerson(char value, LearningExample& le)
  {
    switch(value)
    {
    case '1': le.assoc_context[gcPers1] = 1; break;
    case '2': le.assoc_context[gcPers2] = 1; break;
    case '3': le.assoc_context[gcPers3] = 1; break;
    }
  }
  void encodeDefiniteness(char value, LearningExample& le)
  {
    switch(value)
    {
    case 's': le.assoc_context[gcDefShort] = 1; break;
    case 'f': le.assoc_context[gcDefFull] = 1; break;
    }
  }
  void encodeDegree(char value, LearningExample& le)
  {
    switch(value)
    {
    case 'p': le.assoc_context[gcDegrPos] = 1; break;
    case 'c': le.assoc_context[gcDegrCom] = 1; break;
    case 's': le.assoc_context[gcDegrSup] = 1; break;
    }
  }
  void msd2vec(LearningExample& le, const std::string& msd)
  {
    le.assoc_context.resize(gcLast, 0);
    if (msd.empty()) return;
    if (msd[0] == 'N') // noun
    {
      le.assoc_context[gcPosNoun] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 1 && msd[i] == 'c') le.assoc_context[gcNtCmn] = 1;
        if (i == 1 && msd[i] == 'p') le.assoc_context[gcNtProper] = 1;
        if (i == 2) encodeGender(msd[i], le);
        if (i == 3) encodeNumber(msd[i], le);
        if (i == 4) encodeCase(msd[i], le);
        if (i == 5) encodeAnim(msd[i], le);
      }
    }
    if (msd[0] == 'V') // verb
    {
      le.assoc_context[gcPosVerb] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 2)
        {
          switch (msd[i])
          {
          case 'i': le.assoc_context[gcVfInd] = 1; break;
          case 'm': le.assoc_context[gcVfImp] = 1; break;
          case 'c': le.assoc_context[gcVfCond] = 1; break;
          case 'n': le.assoc_context[gcVfInf] = 1; break;
          case 'p': le.assoc_context[gcVfPart] = 1; break;
          case 'g': le.assoc_context[gcVfGer] = 1; break;
          }
        }
        if (i == 3) encodeTense(msd[i], le);
        if (i == 4) encodePerson(msd[i], le);
        if (i == 5) encodeNumber(msd[i], le);
        if (i == 6) encodeGender(msd[i], le);
        if (i == 7)
        {
          switch (msd[i])
          {
          case 'a': le.assoc_context[gcVoiceAct] = 1; break;
          case 'p': le.assoc_context[gcVoicePass] = 1; break;
          }
        }
        if (i == 8) encodeDefiniteness(msd[i], le);
        if (i == 9)
        {
          switch (msd[i])
          {
          case 'p': le.assoc_context[gcAspProg] = 1; break;
          case 'e': le.assoc_context[gcAspPerf] = 1; break;
          }
        }
        if (i == 10) encodeCase(msd[i], le);
      }
    }
    if (msd[0] == 'A') // adjective
    {
      le.assoc_context[gcPosAdj] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 1 and msd[i] == 's') le.assoc_context[gcPossess] = 1;
        if (i == 2) encodeDegree(msd[i], le);
        if (i == 3) encodeGender(msd[i], le);
        if (i == 4) encodeNumber(msd[i], le);
        if (i == 5) encodeCase(msd[i], le);
        if (i == 6) encodeDefiniteness(msd[i], le);
      }
    }
    if (msd[0] == 'P') // pronoun
    {
      le.assoc_context[gcPosPron] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 1)
        {
          switch (msd[i])
          {
          case 'p': le.assoc_context[gcPrntPers] = 1; break;
          case 'd': le.assoc_context[gcPrntDem] = 1; break;
          case 'i': le.assoc_context[gcPrntIndef] = 1; break;
          case 's': le.assoc_context[gcPossess] = 1; break;
          case 'q': le.assoc_context[gcPrntInterrog] = 1; break;
          case 'r': le.assoc_context[gcPrntRelat] = 1; break;
          case 'x': le.assoc_context[gcPrntReflex] = 1; break;
          case 'z': le.assoc_context[gcPrntNeg] = 1; break;
          case 'n': le.assoc_context[gcPrntNspec] = 1; break;
          }
        }
        if (i == 2) encodePerson(msd[i], le);
        if (i == 3) encodeGender(msd[i], le);
        if (i == 4) encodeNumber(msd[i], le);
        if (i == 5) encodeCase(msd[i], le);
        if (i == 6)
        {
          switch (msd[i]) // todo: попробовать перекодировать в части речи
          {
          case 'n': le.assoc_context[gcStNom] = 1; break;
          case 'a': le.assoc_context[gcStAdj] = 1; break;
          case 'r': le.assoc_context[gcStAdv] = 1; break;
          }
        }
        if (i == 7) encodeAnim(msd[i], le);
      }
    }
    if (msd[0] == 'R') // adverb
    {
      le.assoc_context[gcPosAdv] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 1) encodeDegree(msd[i], le);
      }
    }
    if (msd[0] == 'M') // numeral
    {
      le.assoc_context[gcPosNumeral] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 1)
        {
          switch (msd[i])
          {
          case 'c': le.assoc_context[gcNumeralCard] = 1; break;
          case 'o': le.assoc_context[gcNumeralOrd] = 1; break;
          case 'l': le.assoc_context[gcNumeralCollect] = 1; break;
          }
        }
        if (i == 2) encodeGender(msd[i], le);
        if (i == 3) encodeNumber(msd[i], le);
        if (i == 4) encodeCase(msd[i], le);
      }
    }
    if (msd[0] == 'S') // adposition
    {
      le.assoc_context[gcPosAdpos] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 3) encodeCase(msd[i], le);
      }
    }
    if (msd[0] == 'C') // conjunction
      le.assoc_context[gcPosConj] = 1;
    if (msd[0] == 'Q') // particle
      le.assoc_context[gcPosPart] = 1;
    if (msd[0] == 'I') // interjection
      le.assoc_context[gcPosInter] = 1;
  } // method-end
}; // class-decl-end


#endif /* LEARNING_EXAMPLE_PROVIDER_H_ */
