#ifndef ADD_TOKS_H_
#define ADD_TOKS_H_

#include <memory>
#include <string>
#include <fstream>
#include <iostream>

// Добавление токенов в модель (в район леммы или между леммами, если их несколько)
class AddToks
{
public:
  static void run( const std::string& model_fn, bool useTxtFmt = false )
  {
    // 1. Загружаем модель
    std::vector<std::string> vocab;
    float *embeddings;
    size_t words_count;
    size_t emb_size;
    if ( !load_model(model_fn, useTxtFmt, words_count, emb_size, vocab, embeddings) )
      return;

    // 2. Загружаем информацию о токенах (их отображение в леммы)
    std::map<std::string, std::map<size_t, size_t>> t2l_map;
    std::ifstream t2l_ifs("token_lemmas.map");
    std::string buf;
    while ( std::getline(t2l_ifs, buf).good() )
    {
      std::vector<std::string> parts;
      split_by_space(buf, parts);
      if ( parts.size() < 3 || parts.size() % 2 == 0 ) continue;    // skip invalid records
      std::string token = parts[0];
      // если токен равен хоть какой-нибудь лемме (с уже построенным вектором), то пропускаем его
      size_t dbl_idx = get_word_idx(vocab, token);
      if ( dbl_idx != vocab.size() ) continue;
      // парсим список лемм
      bool isParseOk = true;
      std::map<size_t, size_t> lcmap;
      for ( size_t i = 0; i < ((parts.size()-1)/2); ++i )
      {
        size_t lemma_idx = get_word_idx(vocab, parts[i*2+1]);
        if ( lemma_idx == vocab.size() ) continue;
        std::string cnt_str = parts[i*2+2];
        size_t cnt = 0;
        try { cnt = std::stoi(cnt_str); } catch (...) { isParseOk = false; break; }
        lcmap[lemma_idx] = cnt;
      }
      if (!isParseOk || lcmap.empty()) continue;
      t2l_map[token] = lcmap;
    }

    // 3. Добавляем токены в модель
    //    (за векторное представление токена принимается взвешенное среднее векторов его лемм)
    // сначала выделим память
    float *new_embeddings = (float *) malloc( t2l_map.size() * emb_size * sizeof(float) );
    if (new_embeddings == NULL)
    {
      std::cerr << "Can't allocate memory for new embeddings" << std::endl;
      std::cerr << "    Words: " << t2l_map.size() << std::endl;
      return;
    }
    std::fill(new_embeddings, new_embeddings+emb_size*t2l_map.size(), 0.0);
    // формируем вектора
    float *neOffset = new_embeddings;
    for (auto& token : t2l_map)
    {
      float cnt_sum = 0;
      for (auto& lemma : token.second)
        cnt_sum += lemma.second;
      for (auto& lemma : token.second)
      {
        size_t lemma_idx = lemma.first;
        float weight = (float)lemma.second / cnt_sum;
        float *offset = embeddings + lemma_idx*emb_size;
        for (size_t d = 0; d < emb_size; ++d)
          *(neOffset+d) += *(offset+d) * weight;
      }
      make_embedding_as_neighbour(emb_size, neOffset, neOffset); // немного смещаем получившийся вектор, чтобы все вектора были уникальны
      neOffset += emb_size;
    }

    // 4. Сохраняем модель, расширенную токенами
    FILE *fo = fopen(model_fn.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", vocab.size()+t2l_map.size(), emb_size);
    for (size_t a = 0; a < vocab.size(); ++a)
    {
      fprintf(fo, "%s ", vocab[a].c_str());
      for (size_t b = 0; b < emb_size; ++b)
      {
        if ( !useTxtFmt )
          fwrite(&embeddings[a * emb_size + b], sizeof(float), 1, fo);
        else
          fprintf(fo, " %lf", embeddings[a * emb_size + b]);
      }
      fprintf(fo, "\n");
    }
    neOffset = new_embeddings;
    for (auto& token : t2l_map)
    {
      fprintf(fo, "%s ", token.first.c_str());
      for (size_t b = 0; b < emb_size; ++b)
      {
        if ( !useTxtFmt )
          fwrite(&neOffset[b], sizeof(float), 1, fo);
        else
          fprintf(fo, " %lf", neOffset[b]);
      }
      fprintf(fo, "\n");
      neOffset += emb_size;
    }
    fclose(fo);

  } // method-end

private:
  static bool load_model( const std::string& model_fn, bool useTxtFmt,
                          size_t& words_count, size_t& emb_size, std::vector<std::string>& vocab, float*& embeddings )
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
    vocab.clear();
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
      std::getline(ifs,buf); // считываем конец строки
    }
    ifs.close();
    return true;
  }
  static void split_by_space(const std::string& str, std::vector<std::string>& result)
  {
    size_t prev = 0;
    while (true)
    {
      size_t curr = str.find(' ', prev);
      if (curr == std::string::npos)
      {
        result.push_back( str.substr(prev) );
        break;
      }
      else
      {
        result.push_back( str.substr(prev, curr-prev) );
        prev = curr + 1;
      }
    }
  }
  static size_t get_word_idx(const std::vector<std::string>& vocab, const std::string& word)
  {
    size_t widx = 0;
    for ( ; widx < vocab.size(); ++widx )
      if (vocab[widx] == word)
        break;
    return widx;
  } // method-end

  static void make_embedding_as_neighbour( size_t emb_size, float* base_embedding, float* new_embedding, float distance_factor = 1.0 )
  {
    auto random_sign = []() -> float { return ((float)(rand() % 2) - 0.5)/0.5; };
    for (size_t d = 0; d < emb_size; ++d)
    {
      float *offs = base_embedding + d;
      *(new_embedding + d) = *offs + random_sign() * (*offs / 100 * distance_factor);
    }
  } // method-end

}; // class-decl-end



#endif /* ADD_TOKS_H_ */
