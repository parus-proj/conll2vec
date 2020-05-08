#ifndef SIM_ESTIMATOR_H_
#define SIM_ESTIMATOR_H_

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <numeric>
#include <codecvt>

class SimilarityEstimator
{
public:
  SimilarityEstimator(size_t dep_part, size_t assoc_part)
  : dep_size(dep_part)
  , assoc_size(assoc_part)
  , embeddings(nullptr)
  , words_count(0)
  , emb_size(0)
  , cmp_dims(cdAll)
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
    // в цикле считываем слова и ищем для них ближайшие (по косинусной мере) в векторной модели
    while (true)
    {
      // запрашиваем у пользователя очередное слово
      std::string word;
      std::cout << "Enter word (EXIT to break): ";
      std::cout.flush();
      std::getline(std::cin, word);
      if (word == "EXIT") break;
      if (word == "ALL")   { cmp_dims = cdAll; continue; }
      if (word == "DEP")   { cmp_dims = cdDepOnly; continue; }
      if (word == "ASSOC") { cmp_dims = cdAssocOnly; continue; }
      // ищем слово в словаре (проверим, что оно есть и получим индекс)
      size_t widx = get_word_idx(word);
      if (widx == words_count)
      {
        std::cout << "  out of dictionary word..." << std::endl;
        continue;
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
      std::cout << "                                       word | cosine similarity" << std::endl
                << "  -------------------------------------------------------------" << std::endl;
      for (auto& w : best)
      {
        size_t word_len = To_UTF32(w.second).length();
        std::string alignedWord = (word_len >= 41) ? w.second : (std::string(41-word_len, ' ') + w.second);
        std::cout << "  " << alignedWord << "   " << w.first << std::endl;
      }
    } // infinite loop
  } // method-end
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
  // режим сравнения (какие измерения используются)
  enum CmpDims
  {
    cdAll,
    cdDepOnly,
    cdAssocOnly
  } cmp_dims;

  size_t get_word_idx(const std::string& word)
  {
    size_t widx = 0;
    for ( ; widx < vocab.size(); ++widx )
      if (vocab[widx] == word)
        break;
    return widx;
  } // method-end

  std::u32string To_UTF32(const std::string &s)
  {
      std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
      return conv.from_bytes(s);
  }

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
};


#endif /* SIM_ESTIMATOR_H_ */
