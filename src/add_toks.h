#ifndef ADD_TOKS_H_
#define ADD_TOKS_H_

#include "vectors_model.h"

#include <memory>
#include <string>
#include <fstream>
#include <iostream>

// Добавление токенов в модель (в район леммы или между леммами, если их несколько)
class AddToks
{
public:
  static void run( const std::string& model_fn, const std::string& tlm_fn, bool useTxtFmt = false )
  {
    // 1. Загружаем модель (с леммами)
    VectorsModel vm;
    if ( !vm.load(model_fn, useTxtFmt) )
      return;

    // 2. Загружаем информацию о токенах (их отображение в леммы)
    std::map<std::string, std::map<size_t, size_t>> t2l_map;
    std::ifstream t2l_ifs( tlm_fn.c_str() );
    std::string buf;
    while ( std::getline(t2l_ifs, buf).good() )
    {
      std::vector<std::string> parts;
      split_by_space(buf, parts);
      if ( parts.size() < 3 || parts.size() % 2 == 0 ) continue;    // skip invalid records
      std::string token = parts[0];
      // если токен равен хоть какой-нибудь лемме (с уже построенным вектором), то пропускаем его
      size_t dbl_idx = vm.get_word_idx(token);
      if ( dbl_idx != vm.vocab.size() ) continue;
      // парсим список лемм
      bool isParseOk = true;
      std::map<size_t, size_t> lcmap;
      for ( size_t i = 0; i < ((parts.size()-1)/2); ++i )
      {
        size_t lemma_idx = vm.get_word_idx(parts[i*2+1]);
        if ( lemma_idx == vm.vocab.size() ) continue;
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
    float *new_embeddings = (float *) malloc( t2l_map.size() * vm.emb_size * sizeof(float) );
    if (new_embeddings == NULL)
    {
      std::cerr << "Can't allocate memory for new embeddings" << std::endl;
      std::cerr << "    Words: " << t2l_map.size() << std::endl;
      return;
    }
    std::fill(new_embeddings, new_embeddings+vm.emb_size*t2l_map.size(), 0.0);
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
        float *offset = vm.embeddings + lemma_idx*vm.emb_size;
        for (size_t d = 0; d < vm.emb_size; ++d)
          *(neOffset+d) += *(offset+d) * weight;
      }
      VectorsModel::make_embedding_as_neighbour(vm.emb_size, neOffset, neOffset); // немного смещаем получившийся вектор, чтобы все вектора были уникальны
      neOffset += vm.emb_size;
    }

    // 4. Сохраняем модель, расширенную токенами
    FILE *fo = fopen(model_fn.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", vm.vocab.size()+t2l_map.size(), vm.emb_size);
    for (size_t a = 0; a < vm.vocab.size(); ++a)
    {
      fprintf(fo, "%s ", vm.vocab[a].c_str());
      for (size_t b = 0; b < vm.emb_size; ++b)
      {
        if ( !useTxtFmt )
          fwrite(&vm.embeddings[a * vm.emb_size + b], sizeof(float), 1, fo);
        else
          fprintf(fo, " %lf", vm.embeddings[a * vm.emb_size + b]);
      }
      fprintf(fo, "\n");
    }
    neOffset = new_embeddings;
    for (auto& token : t2l_map)
    {
      fprintf(fo, "%s ", token.first.c_str());
      for (size_t b = 0; b < vm.emb_size; ++b)
      {
        if ( !useTxtFmt )
          fwrite(&neOffset[b], sizeof(float), 1, fo);
        else
          fprintf(fo, " %lf", neOffset[b]);
      }
      fprintf(fo, "\n");
      neOffset += vm.emb_size;
    }
    fclose(fo);
  } // method-end

private:
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
  } // method-end
}; // class-decl-end



#endif /* ADD_TOKS_H_ */
