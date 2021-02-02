#ifndef SIM_ESTIMATOR_H_
#define SIM_ESTIMATOR_H_

#include "str_conv.h"
#include "vectors_model.h"

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <numeric>
#include <optional>
#include <iostream>
#include <fstream>
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
    cdAssocOnly,
    cdGrammOnly
  };
public:
  SimilarityEstimator(size_t dep_part, size_t assoc_part, size_t gramm_part, float assoc_ratio)
  : dep_size(dep_part)
  , assoc_size(assoc_part)
  , gramm_size(gramm_part)
  , a_ratio_sqr(assoc_ratio*assoc_ratio)
  , cmp_dims(cdAll)
  , cmp_mode(cmWord)
  {
  }
  bool load_model(const std::string& model_fn, bool useTxtFmt = false)
  {
    return vm.load(model_fn, useTxtFmt, true);
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
                    "  DIM=GRAMM -- use only 'grammatical' dimensions\n"
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
      if (word == "DIM=GRAMM") { cmp_dims = cdGrammOnly; continue; }
      if (word == "MOD=WORD")  { cmp_mode = cmWord; continue; }
      if (word == "MOD=PAIR")  { cmp_mode = cmPair; continue; }
      switch ( cmp_mode )
      {
      case cmWord: word_mode_helper(word); break;
      case cmPair: pair_mode_helper(word); break;
      }
    } // infinite loop
  } // method-end
  void run_for_file(const std::string& filename, const std::string& fmt)
  {
    std::ifstream ifs(filename);
    if ( !ifs.good() )
    {
      std::cerr << "Can't open file: " << filename << std::endl;
      return;
    }
    std::string line;
    if (fmt == "russe") // process header
    {
      std::getline(ifs, line);
      std::cout << line << std::endl;
    }
    while ( std::getline(ifs, line).good() )
    {
      size_t delimiters_count = std::count(line.begin(), line.end(), ',');
      if (delimiters_count != 2)
      {
        std::cerr << "Invalid record: " << line << std::endl;
        continue;
      }
      size_t c1 = line.find(',');
      size_t c2 = line.find(',', c1+1);
      auto word1 = line.substr(0, c1);
      auto word2 = line.substr(c1+1, c2-(c1+1));
      auto sim = get_sim(cdAll, word1, word2);
      std::cout << word1 << "," << word2 << "," << sim.value_or(0);
      if (fmt == "detail")
      {
        auto simd = get_sim(cdDepOnly, word1, word2);
        auto sima = get_sim(cdAssocOnly, word1, word2);
        std::cout << "," << simd.value_or(0) << "," << sima.value_or(0);
      }
      std::cout << std::endl;
    }
  } // method-end
  // получение расстояния для заданной пары слов
  std::optional<float> get_sim(CmpDims dims, const std::string& word1, const std::string& word2)
  {
    auto widx1 = vm.get_word_idx(word1);
    auto widx2 = vm.get_word_idx(word2);
    if (widx1 == vm.words_count || widx2 == vm.words_count)
      return std::nullopt;
    float* w1Offset = vm.embeddings + widx1*vm.emb_size;
    float* w2Offset = vm.embeddings + widx2*vm.emb_size;
    return cosine_measure(w1Offset, w2Offset, dims);
  }
  // предоставление доступа к векторному пространству
  VectorsModel* raw()
  {
    return &vm;
  }
private:
  size_t dep_size;
  size_t assoc_size;
  size_t gramm_size;
  // вклад ассоциативной части вектора в оценку близости (относительно категориальной части)
  float a_ratio_sqr;
  // контейнер векторной модели
  VectorsModel vm;
  // измерения для сравнения
  CmpDims cmp_dims;
  // режим сравнения
  enum CmpMode
  {
    cmWord,     // вывод соседей указанного слова
    cmPair      // оценка сходства между парой слов
  } cmp_mode;

  float cosine_measure(float* w1_Offset, float* w2_Offset, CmpDims dims)
  {
    float result = 0;
    size_t assoc_begin = dep_size;
    size_t gramm_begin = dep_size + assoc_size;
    switch ( dims )
    {
    case cdAll       : //result = std::inner_product(w1_Offset, w1_Offset+vm.emb_size, w2_Offset, 0.0); break;
                       {
                         float l1 = 0, l2 = 0;
                         for (size_t i = 0; i < vm.emb_size; ++i)
                         {
                           if (i < assoc_begin)
                           {
                             result += w1_Offset[i]*w2_Offset[i];
                             l1 += w1_Offset[i]*w1_Offset[i];
                             l2 += w2_Offset[i]*w2_Offset[i];
                           }
                           else if (i < gramm_begin)
                           {
                             result += w1_Offset[i]*w2_Offset[i] * a_ratio_sqr;
                             l1 += w1_Offset[i]*w1_Offset[i] * a_ratio_sqr;
                             l2 += w2_Offset[i]*w2_Offset[i] * a_ratio_sqr;
                           }
                           else
                           {
                             result += w1_Offset[i]*w2_Offset[i];
                             l1 += w1_Offset[i]*w1_Offset[i];
                             l2 += w2_Offset[i]*w2_Offset[i];
                           }
                         }
                         result = result / std::sqrt(l1) / std::sqrt(l2);
                       }
                       break;
    case cdDepOnly   : {
                         if (dep_size == 0) return 0;
                         result = std::inner_product(w1_Offset, w1_Offset+dep_size, w2_Offset, 0.0);
                         result /= std::sqrt( std::inner_product(w1_Offset, w1_Offset+dep_size, w1_Offset, 0.0) );
                         result /= std::sqrt( std::inner_product(w2_Offset, w2_Offset+dep_size, w2_Offset, 0.0) );
                       }
                       break;
    case cdAssocOnly : {
                         if (assoc_size == 0) return 0;
                         float* w1_Offset_fit = w1_Offset + assoc_begin;
                         float* w2_Offset_fit = w2_Offset + assoc_begin;
                         result = std::inner_product(w1_Offset_fit, w1_Offset_fit+assoc_size, w2_Offset_fit, 0.0);
                         result /= std::sqrt( std::inner_product(w1_Offset_fit, w1_Offset_fit+assoc_size, w1_Offset_fit, 0.0) );
                         result /= std::sqrt( std::inner_product(w2_Offset_fit, w2_Offset_fit+assoc_size, w2_Offset_fit, 0.0) );
                       }
                       break;
    case cdGrammOnly : {
                         if (gramm_size == 0) return 0;
                         float* w1_Offset_fit = w1_Offset + gramm_begin;
                         float* w2_Offset_fit = w2_Offset + gramm_begin;
                         result = std::inner_product(w1_Offset_fit, w1_Offset_fit+gramm_size, w2_Offset_fit, 0.0);
                         result /= std::sqrt( std::inner_product(w1_Offset_fit, w1_Offset_fit+gramm_size, w1_Offset_fit, 0.0) );
                         result /= std::sqrt( std::inner_product(w2_Offset_fit, w2_Offset_fit+gramm_size, w2_Offset_fit, 0.0) );
                       }
                       break;
    }
    return result;
  }

  void word_mode_helper(const std::string& word)
  {
    // ищем слово в словаре (проверим, что оно есть и получим индекс)
    size_t widx = vm.get_word_idx(word);
    if (widx == vm.words_count)
    {
      str_to_console( "  out of vocabulary word...\n" );
      return;
    }
    // ищем n ближайших к указанному слову
    float* wiOffset = vm.embeddings + widx*vm.emb_size;
    std::multimap<float, std::string, std::greater<float>> best;
    for (size_t i = 0; i < vm.words_count; ++i)
    {
      if (i == widx) continue;
      float* iOffset = vm.embeddings + i*vm.emb_size;
      float sim = cosine_measure(iOffset, wiOffset, cmp_dims);
      if (best.size() < 40)
        best.insert( std::make_pair(sim, vm.vocab[i]) );
      else
      {
        auto minIt = std::prev( best.end() );
        if (sim > minIt->first)
        {
          best.erase(minIt);
          best.insert( std::make_pair(sim, vm.vocab[i]) );
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
    size_t widx1 = vm.get_word_idx(word1);
    if (widx1 == vm.words_count)
    {
      str_to_console( "  first word is out of vocabulary...\n" );
      return;
    }
    size_t widx2 = vm.get_word_idx(word2);
    if (widx2 == vm.words_count)
    {
      str_to_console( "  second word is out of vocabulary...\n" );
      return;
    }
    // оцениваем и выводим меру близости
    float* w1Offset = vm.embeddings + widx1*vm.emb_size;
    float* w2Offset = vm.embeddings + widx2*vm.emb_size;
    float sim = cosine_measure(w1Offset, w2Offset, cmp_dims);
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
