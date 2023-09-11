#ifndef ADD_TOKS_H_
#define ADD_TOKS_H_

#include "vectors_model.h"
#include "str_conv.h"
#include "special_toks.h"

#include <memory>
#include <string>
#include <fstream>
#include <iostream>

// Добавление токенов в модель (в район леммы или между леммами, если их несколько)
class AddToks
{
public:
  static void run( const std::string& model_fn, const std::string& tlm_fn )
  {
    std::string buf;

    // 1. Загружаем модель (с леммами)
    VectorsModel vm;
    if ( !vm.load(model_fn) )
      return;


    // 2. Загружаем информацию о токенах (их отображение в леммы)
    std::map<std::string, std::map<size_t, size_t>> t2l_map;
    std::ifstream t2l_ifs( tlm_fn.c_str() );
    while ( std::getline(t2l_ifs, buf).good() )
    {
      std::vector<std::string> parts;
      StrUtil::split_by_space(buf, parts);
      if ( parts.size() < 3 || parts.size() % 2 == 0 ) continue;    // skip invalid records
      std::string token = parts[0];
      // если токен равен хоть какой-нибудь лемме (с уже построенным вектором), то пропускаем его
      //   fail: так нельзя: автотехника = (автотехника, автотехник), банка = (банка, банк)
      //size_t dbl_idx = vm.get_word_idx(token);
      //if ( dbl_idx != vm.vocab.size() ) continue;
      // парсим список лемм
      bool isParseOk = true;
      std::map<size_t, size_t> lcmap;
      for ( size_t i = 0; i < ((parts.size()-1)/2); ++i )
      {
        const auto& lemma = parts[i*2+1];
        const auto& cnt_str = parts[i*2+2];
        size_t lemma_idx = vm.get_word_idx_fast(lemma);
        if ( lemma_idx == vm.vocab.size() ) continue;
        size_t cnt = 0;
        try { cnt = std::stoi(cnt_str); } catch (...) { isParseOk = false; break; }
        // проверим минимальную представительность токена леммой
        // иначе возникают ситуации (нельзя-с нельзя-с 97 с 37)
        // здесь "нельзя-с" недобирает частотой до леммы, а "с" частотно; получается токен "нельзя-с" формируется равным "с"
        if (cnt < 50) continue;
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
      /////
      rebalance( token.second ); // ограничим вклад наиболее частотной леммы (размажем вектор между леммами)
      /////
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

    // 4. Сохраняем модель токенов
    FILE *fo = fopen(model_fn.c_str(), "wb");
    fprintf(fo, "%lu %lu %lu %lu %lu\n", t2l_map.size(), vm.emb_size, vm.dep_size, vm.assoc_size, (long unsigned int)0);
    neOffset = new_embeddings;
    for (auto& token : t2l_map)
    {
      VectorsModel::write_embedding(fo, token.first, neOffset, vm.emb_size);
      neOffset += vm.emb_size;
    }

    fclose(fo);
  } // method-end

private:
  static void rebalance(std::map<size_t, size_t>& lcmap)
  {
    if ( lcmap.size() < 2 ) return;
    // перебалансировка частот лемм (ограничение сверху в 60%)
    auto it = lcmap.begin();
    auto maxIt = it;
    size_t cnt_sum = lcmap.begin()->second;
    for (++it; it != lcmap.end(); ++it)
    {
      cnt_sum += it->second;
      if ( it->second > maxIt->second )
        maxIt = it;
    }
    size_t cnt_sum_without_max = cnt_sum - maxIt->second; // по этому количеству будем считать доли в перераспределении

    float maxRate = (float)(maxIt->second) / (float)cnt_sum;
    if (maxRate <= 0.6) return;

    size_t new_max_cnt = static_cast<size_t>(0.6 * cnt_sum); // todo: округление к целому
    size_t delta = maxIt->second - new_max_cnt; // это та величина, которую надо перераспределить на другие леммы
    if (delta == 0) return;

    // надо выбрать всю delta, а если останется, вернуть остаток в maxIt->second
    size_t delta_tail = delta;
    for (auto it = lcmap.begin(); it != lcmap.end(); ++it)
    {
      if ( it == maxIt ) continue;
      float itRate = (float)(it->second) / (float)cnt_sum_without_max;
      size_t addon = static_cast<size_t>(itRate * delta); // todo: округление к целому
      delta_tail -= addon;
      it->second += addon;

    }
    maxIt->second = new_max_cnt + delta_tail;

    // todo: считаем снова сумму для контроля (она не должна измениться)
    size_t check_sum = 0;
    for (auto it = lcmap.begin(); it != lcmap.end(); ++it)
      check_sum += it->second;
    if (check_sum != cnt_sum)
    {
      std::cerr << "Rebalance checksum error!" << std::endl;
      throw std::runtime_error("rebalance");
    }
  }

}; // class-decl-end



#endif /* ADD_TOKS_H_ */
