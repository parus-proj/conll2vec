#ifndef UNPNIZER_H_
#define UNPNIZER_H_

#include "original_word2vec_vocabulary.h"
#include "str_conv.h"

#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>

// Средство для слияния основного словаря со словарем собственных имен:
// - удаляются суффиксы _PN;
// - при совпадении форм
//   либо выбирается вектор, соответствующий наиболее частотному варианту (выбор между нарицательным и собственным)
//   либо вычисляется их взвешенное среднее (средний вектор)
// Используется для создания модели слов без мета-суффиксов и для подавления ошибок в conll-разметке (в части признака собственное/нарицательное).
class Unpnizer
{
public:
  static void run( std::shared_ptr< OriginalWord2VecVocabulary> mainVocabulary,
                   std::shared_ptr< OriginalWord2VecVocabulary> properVocabulary,
                   const std::string& model_fn, bool useTxtFmt = false )
  {
    // 1. Загружаем модель
    std::vector<std::u32string> vocab;
    float *embeddings;
    size_t words_count;
    size_t emb_size;
    // открываем файл модели
    std::ifstream ifs(model_fn.c_str(), std::ios::binary);
    if ( !ifs.good() )
    {
      std::cerr << "Model file not found" << std::endl;
      return;
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
      return;
    }
    // загрузка словаря и векторов
    vocab.reserve(words_count);
    for (uint64_t w = 0; w < words_count; ++w)
    {
      std::getline(ifs, buf, ' '); // читаем слово (до пробела)
      vocab.push_back(StrConv::To_UTF32(buf));
      // читаем вектор
      float* eOffset = embeddings + w*emb_size;
      if ( !useTxtFmt )
        ifs.read( reinterpret_cast<char*>( eOffset ), sizeof(float)*emb_size );
      else
      {
        for (size_t j = 0; j < emb_size; ++j)
          ifs >> eOffset[j];
      }
      std::getline(ifs,buf); // считываем конец строки
    }
    ifs.close();

    // 2. Редуцируем модель
    for (size_t vidx = 0; vidx < vocab.size(); ++vidx)
    {
      auto sfx_pos = vocab[vidx].find(U"_PN");
      if ( sfx_pos != std::u32string::npos &&  sfx_pos == vocab[vidx].length()-3 )
      {
        std::u32string pair_word = vocab[vidx].substr(0, vocab[vidx].length()-3);
        size_t pair_word_idx = get_word_idx(vocab, pair_word);
        if (pair_word_idx == vocab.size())
          vocab[vidx] = pair_word; // если парного нет, то всего-лишь отсекаем суффикс _PN
        else
        {
          float* pn_word_offset = embeddings + vidx*emb_size;
          float* pair_word_offset = embeddings + pair_word_idx*emb_size;
          auto pn_word_cn = properVocabulary->idx_to_data( properVocabulary->word_to_idx(StrConv::To_UTF8(pair_word)) ).cn; // в словаре хранятся без суффиксов
          auto pair_word_cn = mainVocabulary->idx_to_data( mainVocabulary->word_to_idx(StrConv::To_UTF8(pair_word)) ).cn;
          // сценарий редукции к наиболее частотному
          //select_most_frequent(vocab, vidx, pair_word_idx, pn_word_offset, pair_word_offset, pn_word_cn, pair_word_cn);
          // сценарий редукции взвешенным средним вектором
          select_by_ratio(vocab, vidx, pair_word_idx, pn_word_offset, pair_word_offset, emb_size, pn_word_cn, pair_word_cn);
        }
      }
    }

    // 3. Сохраняем сжатую модель
    size_t new_vocab_size = std::count_if(vocab.begin(), vocab.end(), [](const std::u32string& item) {return !item.empty();});
    FILE *fo = fopen(model_fn.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", new_vocab_size, emb_size);
    for (size_t a = 0; a < vocab.size(); ++a)
    {
      if ( vocab[a].empty() )
        continue;
      fprintf(fo, "%s ", StrConv::To_UTF8(vocab[a]).c_str());
      for (size_t b = 0; b < emb_size; ++b)
      {
        if ( !useTxtFmt )
          fwrite(&embeddings[a * emb_size + b], sizeof(float), 1, fo);
        else
          fprintf(fo, " %lf", embeddings[a * emb_size + b]);
      }
      fprintf(fo, "\n");
    }
    fclose(fo);
  } // method-end
private:
  static size_t get_word_idx(const std::vector<std::u32string>& vocab, const std::u32string& word)
  {
    size_t widx = 0;
    for ( ; widx < vocab.size(); ++widx )
      if (vocab[widx] == word)
        break;
    return widx;
  } // method-end

  static void select_most_frequent( std::vector<std::u32string>& vocab, size_t pn_word_idx, size_t pair_word_idx,
                                    float* pn_word_offset, float* pair_word_offset,
                                    uint64_t pn_word_cn, uint64_t pair_word_cn)
  {
    if (pn_word_cn > pair_word_cn)
    {
      vocab[pn_word_idx] = vocab[pair_word_idx];
      vocab[pair_word_idx].clear();
    }
    else
    {
      vocab[pn_word_idx].clear();
    }
  } // method-end

  static void select_by_ratio( std::vector<std::u32string>& vocab, size_t pn_word_idx, size_t pair_word_idx,
                               float* pn_word_offset, float* pair_word_offset, size_t emb_size,
                               uint64_t pn_word_cn, uint64_t pair_word_cn)
  {
    float pn_fraction = (float)pn_word_cn / (float)(pn_word_cn + pair_word_cn);
    vocab[pn_word_idx].clear();
    for (size_t i = 0; i < emb_size; ++i)
      pair_word_offset[i] = pn_fraction * pn_word_offset[i] + (1.0 - pn_fraction) * pair_word_offset[i];

    for (size_t i = 0; i < emb_size; ++i)
      if ( !std::isnormal(pair_word_offset[i]) )
      {
        std::cerr << StrConv::To_UTF8(vocab[pair_word_idx]) << " -- " << pn_fraction << ", " << pn_word_cn << ", " << pair_word_cn << std::endl;
        break;
      }
  } // method-end

};


#endif /* UNPNIZER_H_ */
