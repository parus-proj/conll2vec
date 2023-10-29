#ifndef MAKE_RUE_EMBEDDINGS_H_
#define MAKE_RUE_EMBEDDINGS_H_

#include "vectors_model.h"
#include "str_conv.h"
#include "add_punct.h"

#include <memory>
#include <string>
#include <fstream>
#include <iostream>

// Построение мэппингов и базовых эмбеддингов для RUE-модели
class MakeRueEmbeddings
{
public:
  static void run( const std::string& model_fn, const std::string& tlm_fn )
  {
    std::string buf;

    // 1. Загружаем модель (с леммами)
    VectorsModel vm;
    if ( !vm.load(model_fn) )
      return;
    // 1.1. Расшираем модель знаками препинания
    if ( !AddPunct::merge_punct(vm) )
      return;
    const std::set<std::string> puncts =    { ".", ",", "!", "?", ";", "…", "...",
                                              ":", "-", "--", "—", "–", "‒",
                                              "'", "ʼ", "ˮ", "\"",
                                              "«", "“", "„", "‘", "‚",
                                              "»", "”", "‟", "’", "‛",
                                              "(", "[", "{", "⟨",
                                              ")", "]", "}", "⟩"
                                            };


    // 2. Загружаем информацию о токенах (их отображение в леммы)
    std::map<std::string, std::map<size_t, size_t>> t2l_map;
    std::ifstream t2l_ifs( tlm_fn.c_str() );
    while ( std::getline(t2l_ifs, buf).good() )
    {
      std::vector<std::string> parts;
      StrUtil::split_by_space(buf, parts);
      if ( parts.size() < 3 || parts.size() % 2 == 0 ) continue;    // skip invalid records
      std::string token = parts[0];
      if ( puncts.find(token) != puncts.end() ) continue;
      // если токен равен хоть какой-нибудь лемме (с уже построенным вектором), то пропускаем его
      //   fail: так нельзя: автотехника = (автотехника, автотехник), банка = (банка, банк)
      //   size_t dbl_idx = vm.get_word_idx(token);
      //   if ( dbl_idx != vm.vocab.size() ) continue;
      // парсим список лемм
      bool isParseOk = true;
      std::map<size_t, size_t> lcmap;
      for ( size_t i = 0; i < ((parts.size()-1)/2); ++i )
      {
        const auto& lemma = parts[i*2+1];
        if ( puncts.find(lemma) != puncts.end() ) continue;
        const auto& cnt_str = parts[i*2+2];
        size_t lemma_idx = vm.get_word_idx_fast(lemma);
        if ( lemma_idx == vm.words_count ) continue;
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

    // 3. Формируем векторное пространство для токенов и мэппинг (токен_строка -> индекс_эмбеддинга)
    //    (за векторное представление токена принимается взвешенное среднее векторов его лемм)
    // сначала выделим память
    const size_t tokens_count = t2l_map.size();
    const size_t max_embs_count = tokens_count + puncts.size();
    float *new_embeddings = (float *) malloc( max_embs_count * vm.emb_size * sizeof(float) );
    if (new_embeddings == NULL)
    {
      std::cerr << "Can't allocate memory for new embeddings" << std::endl;
      std::cerr << "    Words: " << max_embs_count << std::endl;
      return;
    }
    std::fill(new_embeddings, new_embeddings+vm.emb_size*max_embs_count, 0.0);
    // формируем вектора
    std::map<std::string, size_t> t2e_map; // отображение токенов в индексы эмбеддингов
    std::map<size_t, size_t> l2e_map;      // отображение лемм в индексы эмбеддингов
    size_t embs_count = 0;
    float *neOffset = new_embeddings;
    for (auto& token : t2l_map)
    {
      auto& lcmap = token.second; // отображение лемма->количество (сколько раз токен соотносился с каждой леммой)
      if (lcmap.size() == 1)
      {
        // если токен относится только к одной лемме, то пытаемся "уплотнить" матрицу эмбеддингов
        const size_t lemma_idx = lcmap.begin()->first;
        const auto l2eIt = l2e_map.find(lemma_idx);
        if ( l2eIt != l2e_map.end() )
        {
          const size_t embIdx = l2eIt->second;
          t2e_map[token.first] = embIdx;
        } else {
          float *offset = vm.embeddings + lemma_idx*vm.emb_size;
          std::memcpy(neOffset, offset, vm.emb_size*sizeof(float));
          neOffset += vm.emb_size;
          t2e_map[token.first] = embs_count;
          l2e_map[lemma_idx] = embs_count;
          ++embs_count;
        }
        continue;
      }
      /////
      rebalance( lcmap ); // ограничим вклад наиболее частотной леммы (размажем вектор между леммами)
      /////
      float cnt_sum = 0;
      for (auto& lemma : lcmap)
        cnt_sum += lemma.second;
      for (auto& lemma : lcmap)
      {
        size_t lemma_idx = lemma.first;
        float weight = (float)lemma.second / cnt_sum;
        float *offset = vm.embeddings + lemma_idx*vm.emb_size;
        for (size_t d = 0; d < vm.emb_size; ++d)
          *(neOffset+d) += *(offset+d) * weight;
      }
      neOffset += vm.emb_size;
      t2e_map[token.first] = embs_count;
      ++embs_count;
    }

    // 4. Допишем знаки препинания (их нет в отображении токены->леммы)
    for (const auto& p : puncts)
    {
      size_t idx = vm.get_word_idx_fast(p);
      float *offset = vm.embeddings + idx*vm.emb_size;
      std::memcpy(neOffset, offset, vm.emb_size*sizeof(float));
      neOffset += vm.emb_size;
      t2e_map[p] = embs_count;
      ++embs_count;
    }

    // 5. Сохраняем модель токенов
    const std::string toks_fn = model_fn + ".lex";
    FILE *fo = fopen(toks_fn.c_str(), "wb");
    fprintf(fo, "%lu %lu %lu %lu\n", embs_count, vm.emb_size, vm.dep_size, vm.assoc_size);
    for (size_t i = 0; i < embs_count; ++i)
    {
      VectorsModel::write_embedding__vec(fo, new_embeddings + i * vm.emb_size, 0, vm.emb_size);
    }
    fclose(fo);

    // 6. Сохраняем мэппинг (токен_строка -> индекс_эмбеддинга)
    const std::string t2e_fn = model_fn + ".t2lex";
    std::ofstream t2e_ofs(t2e_fn);
    for (const auto& [token_str, embedding_idx] : t2e_map)
    {
      t2e_ofs << token_str << " " << embedding_idx << std::endl;
    }

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
    if (maxRate <= 0.7) return;

    size_t new_max_cnt = static_cast<size_t>(0.7 * cnt_sum); // todo: округление к целому
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



#endif /* MAKE_RUE_EMBEDDINGS_H_ */
