#ifndef SIM_ESTIMATOR_H_
#define SIM_ESTIMATOR_H_

#include "str_conv.h"

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <numeric>
#include <optional>
#include <iostream>
#ifdef _MSC_VER
  #include <io.h>
  #include <fcntl.h>
#endif

class SimilarityEstimator
{
public:
  // измерения для сравнения
  enum CmpDims
  {
    cdAll,
    cdDepOnly,
    cdAssocOnly
  };
public:
  SimilarityEstimator(size_t dep_part, size_t assoc_part)
  : dep_size(dep_part)
  , assoc_size(assoc_part)
  , embeddings(nullptr)
  , words_count(0)
  , emb_size(0)
  , cmp_dims(cdAll)
  , cmp_mode(cmWord)
  {
  }
  bool load_model(const std::string& model_fn, bool useTxtFmt = false)
  {
    // открываем файл модели
    std::ifstream ifs(model_fn.c_str(), std::ios::binary);
    if ( !ifs.good() )
    {
      std::cerr << "Model file not found" << std::endl;
      return false;
    }
    std::string buf;
    // считыавем размер матрицы
    ifs >> words_count;
    ifs >> emb_size;
    std::getline(ifs,buf); // считываем конец строки
    // выделяем память для эмбеддингов
    embeddings = (float *) malloc( words_count * emb_size * sizeof(float) );
    if (embeddings == NULL)
    {
      std::cerr << "Cannot allocate memory: " << (words_count * emb_size * sizeof(float) / 1048576) << " MB" << std::endl;
      std::cerr << "    Words: " << words_count << std::endl;
      std::cerr << "    Embedding size: " << emb_size << std::endl;
      return false;
    }
    // загрузка словаря и векторов
    vocab.reserve(words_count);
    for (uint64_t w = 0; w < words_count; ++w)
    {
      std::getline(ifs, buf, ' '); // читаем слово (до пробела)
      vocab.push_back(buf);
      // читаем вектор
      float* eOffset = embeddings + w*emb_size;
      if ( !useTxtFmt )
        ifs.read( reinterpret_cast<char*>( eOffset ), sizeof(float)*emb_size );
      else
      {
        for (size_t j = 0; j < emb_size; ++j)
          ifs >> eOffset[j];
      }
      // нормируем вектор (все компоненты в диапазон [-1; +1]
      float len = std::sqrt( std::inner_product(eOffset, eOffset+emb_size, eOffset, 0.0) );
      if (len == 0)
      {
        std::cerr << "Embedding normalization error: Division by zero" << std::endl;
        free(embeddings);
        return false;
      }
      std::transform(eOffset, eOffset+emb_size, eOffset, [len](float a) -> float {return a/len;});
      std::getline(ifs,buf); // считываем конец строки
    }
    return true;
  } // method-end
  void run()
  {
    #ifdef _MSC_VER
      _setmode( _fileno(stdout), _O_U16TEXT );
      _setmode( _fileno(stdin), _O_U16TEXT );
    #endif
    // выводим подсказку
    str_to_console( "\nCOMMANDS: \n"
                    "  EXIT -- terminate this program\n"
                    "  DIM=ALL -- use all dimensions to calculate similarity\n"
                    "  DIM=DEP -- use only 'dependency' dimensions\n"
                    "  DIM=ASSOC -- use only 'associative' dimensions\n"
                    "  MOD=WORD -- find most similar words for selected one\n"
                    "  MOD=PAIR -- estimate similarity for pair of words\n"
                    "Initially: DIM=ALL, MOD=WORD\n\n" );
    // в цикле считываем слова и ищем для них ближайшие (по косинусной мере) в векторной модели
    while (true)
    {
      // запрашиваем у пользователя очередное слово
      str_to_console( "Enter word (EXIT to break): " );
      std::string word = str_from_console();
      if (word == "EXIT") break;
      if (word == "DIM=ALL")   { cmp_dims = cdAll; continue; }
      if (word == "DIM=DEP")   { cmp_dims = cdDepOnly; continue; }
      if (word == "DIM=ASSOC") { cmp_dims = cdAssocOnly; continue; }
      if (word == "MOD=WORD")  { cmp_mode = cmWord; continue; }
      if (word == "MOD=PAIR")  { cmp_mode = cmPair; continue; }
      switch ( cmp_mode )
      {
      case cmWord: word_mode_helper(word); break;
      case cmPair: pair_mode_helper(word); break;
      }
    } // infinite loop
  } // method-end
  // получение расстояния для заданной пары слов
  std::optional<float> get_sim(CmpDims dims, const std::string& word1, const std::string& word2)
  {
    auto widx1 = get_word_idx(word1);
    auto widx2 = get_word_idx(word2);
    if (widx1 == words_count || widx2 == words_count)
      return std::nullopt;
    cmp_dims = dims;
    float* w1Offset = embeddings + widx1*emb_size;
    float* w2Offset = embeddings + widx2*emb_size;
    return cosine_measure(w1Offset, w2Offset);
  }
  // предоставление доступа к векторному пространству
  void raw(size_t& wordsCnt, size_t& embSize, float*& vectors, std::vector<std::string>& words)
  {
    wordsCnt = words_count;
    embSize = emb_size;
    vectors = embeddings;
    words = vocab;
  }
private:
  size_t dep_size;
  size_t assoc_size;
  // контейнер для словаря модели
  std::vector<std::string> vocab;
  // хранилище для векторов
  float *embeddings;
  // параметры модели
  size_t words_count;
  size_t emb_size;
  // измерения для сравнения
  CmpDims cmp_dims;
  // режим сравнения
  enum CmpMode
  {
    cmWord,     // вывод соседей указанного слова
    cmPair      // оценка сходства между парой слов
  } cmp_mode;

  size_t get_word_idx(const std::string& word)
  {
    size_t widx = 0;
    for ( ; widx < vocab.size(); ++widx )
      if (vocab[widx] == word)
        break;
    return widx;
  } // method-end

  float cosine_measure(float* w1_Offset, float* w2_Offset)
  {
    float result = 0;
    switch ( cmp_dims )
    {
    case cdAll       : result = std::inner_product(w1_Offset, w1_Offset+emb_size, w2_Offset, 0.0); break;
    case cdDepOnly   : result = std::inner_product(w1_Offset, w1_Offset+dep_size, w2_Offset, 0.0);
                       result /= std::sqrt( std::inner_product(w1_Offset, w1_Offset+dep_size, w1_Offset, 0.0) );
                       result /= std::sqrt( std::inner_product(w2_Offset, w2_Offset+dep_size, w2_Offset, 0.0) );
                       break;
    case cdAssocOnly : result = std::inner_product(w1_Offset+dep_size, w1_Offset+emb_size, w2_Offset+dep_size, 0.0);
                       result /= std::sqrt( std::inner_product(w1_Offset+dep_size, w1_Offset+emb_size, w1_Offset+dep_size, 0.0) );
                       result /= std::sqrt( std::inner_product(w2_Offset+dep_size, w2_Offset+emb_size, w2_Offset+dep_size, 0.0) );
                       break;
    }
    return result;
  }

  void word_mode_helper(const std::string& word)
  {
    // ищем слово в словаре (проверим, что оно есть и получим индекс)
    size_t widx = get_word_idx(word);
    if (widx == words_count)
    {
      str_to_console( "  out of vocabulary word...\n" );
      return;
    }
    // ищем n ближайших к указанному слову
    float* wiOffset = embeddings + widx*emb_size;
    std::multimap<float, std::string, std::greater<float>> best;
    for (size_t i = 0; i < words_count; ++i)
    {
      if (i == widx) continue;
      float* iOffset = embeddings + i*emb_size;
      float sim = cosine_measure(iOffset, wiOffset);
      if (best.size() < 40)
        best.insert( std::make_pair(sim, vocab[i]) );
      else
      {
        auto minIt = std::prev( best.end() );
        if (sim > minIt->first)
        {
          best.erase(minIt);
          best.insert( std::make_pair(sim, vocab[i]) );
        }
      }
    }
    // выводим результат поиска
    str_to_console( "                                       word | cosine similarity\n"
                    "  -------------------------------------------------------------\n" );
    for (auto& w : best)
    {
      size_t word_len = StrConv::To_UTF32(w.second).length();
      std::string alignedWord = (word_len >= 41) ? w.second : (std::string(41-word_len, ' ') + w.second);
      str_to_console( "  " + alignedWord + "   " + std::to_string(w.first) + "\n" );
    }
  } // method-end

  void pair_mode_helper(const std::string& word1)
  {
    // запрашиваем у пользователя второе слово
    std::string word2;
    str_to_console( "Enter second word: " );
    word2 = str_from_console();
    // ищем слова в словаре
    size_t widx1 = get_word_idx(word1);
    if (widx1 == words_count)
    {
      str_to_console( "  first word is out of vocabulary...\n" );
      return;
    }
    size_t widx2 = get_word_idx(word2);
    if (widx2 == words_count)
    {
      str_to_console( "  second word is out of vocabulary...\n" );
      return;
    }
    // оцениваем и выводим меру близости
    float* w1Offset = embeddings + widx1*emb_size;
    float* w2Offset = embeddings + widx2*emb_size;
    float sim = cosine_measure(w1Offset, w2Offset);
    str_to_console( "cosine similarity = " + std::to_string(sim) + "\n" );
  }

  // вывод в консоль
  void str_to_console(const std::string& str)
  {
    #ifdef _MSC_VER
      static std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
      std::wstring wide_str = converter.from_bytes(str);
      std::wcout << wide_str;
      std::wcout.flush();
    #else
      std::cout << str;
      std::cout.flush();
    #endif
  }
  // ввод строки из консоли
  std::string str_from_console()
  {
    #ifdef _MSC_VER
      std::wstring line;
      std::getline(std::wcin, line);
      static std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
      return converter.to_bytes(line);
    #else
      std::string line;
      std::getline(std::cin, line);
      return line;
    #endif
  }
};


#endif /* SIM_ESTIMATOR_H_ */
