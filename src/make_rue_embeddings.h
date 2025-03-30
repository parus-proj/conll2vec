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
  void run( const std::string& model_fn, const std::string& tlm_fn )
  {
    std::string buf;

    // 1. Загружаем модель (с леммами)
    VectorsModel vm;
    if ( !vm.load(model_fn) )
      return;
    lemmas_model_size = vm.words_count; // for dbg output
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
      ++toks_cnt; // for dbg output
      std::vector<std::string> parts;
      StrUtil::split_by_space(buf, parts);
      if ( parts.size() < 3 || parts.size() % 2 == 0 ) { ++skipped_toks_cnt; continue; }    // skip invalid records
      std::string token = parts[0];
      if ( puncts.find(token) != puncts.end() ) { ++skipped_toks_cnt; continue; }
      // если токен равен хоть какой-нибудь лемме (с уже построенным вектором), то пропускаем его
      //   fail: так нельзя: автотехника = (автотехника, автотехник), банка = (банка, банк)
      //   size_t dbl_idx = vm.get_word_idx(token);
      //   if ( dbl_idx != vm.vocab.size() ) continue;
      // парсим список лемм
      bool isParseOk = true;
      std::map<size_t, size_t> lcmap; // мэппинг из индекса лемммы в частоту (в составе текущего токена)
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
      if ( !isParseOk ) { ++skipped_toks_cnt; continue; }
      if ( lcmap.empty() )
      { 
        ++skipped_toks_cnt; // for-debug-output
        ++skipped_toks_no_lemmas_cnt; // for-debug-output
        if ( skipped_toks_no_lemmas_examples.size() < 10 ) // for-debug-output
          skipped_toks_no_lemmas_examples.insert(token); // for-debug-output
        continue; 
      }
      t2l_map[token] = lcmap;
    }
    for (const auto&[tk, lc] : t2l_map) // for-debug-output
      for (const auto&[idx, cnt] : lc)
        used_lemmas.insert(idx);
    for (size_t i = 0; i < vm.words_count; ++i) // for-debug-output
      if (used_lemmas.find(i) == used_lemmas.end())
      {
        unused_lemmas_examples.insert(vm.vocab[i]);
        if (unused_lemmas_examples.size() > 9) break;
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
          ++clear_embs_cnt;
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
      ++mixed_embs_cnt; // for-debug-output
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

    dbg_output();

  } // method-end

private:
  // -- статистика (для отладочного вывода)
  // размер модели лемм
  size_t lemmas_model_size = 0;
  // леммы, участвующие в формировании эмбеддингов хотя бы одного токена
  std::set<size_t> used_lemmas;
  // примеры лемм, не участвующих в формировании эмбеддингов
  std::set<std::string> unused_lemmas_examples;
  // количество записей в отображении tokens2lemmas
  size_t toks_cnt = 0;
  // количество записей в tokens2lemmas, которые были пропущены по тем или иным причинам
  size_t skipped_toks_cnt = 0;
  // количество записей в tokens2lemmas, которые были пропущены из-за отсутствия опорных лемм с частотностью для данного токена выше порога представительности
  size_t skipped_toks_no_lemmas_cnt = 0;
  // примеры токенов, пропущенных из-за отсутствия опорных лемм
  std::set<std::string> skipped_toks_no_lemmas_examples;
  // количество представлений, основанных на представлении единстввенной леммы
  size_t clear_embs_cnt = 0;
  // количество представлений, сфорированных как смесь значений лемм ("омонимы")
  size_t mixed_embs_cnt = 0;

  void dbg_output()
  {
    std::cout << "Lemmas model size: " << lemmas_model_size << std::endl;
    std::cout << "    Used lemmas: " << used_lemmas.size() << std::endl;
    std::cout << "    Unused lemmas: " << lemmas_model_size - used_lemmas.size() << std::endl;
    std::string unused_lemmas_str;
    for (const auto w : unused_lemmas_examples)
      unused_lemmas_str += std::string(" ") + w;
    std::cout << "    Unused lemmas examples:" << unused_lemmas_str << std::endl;
    std::cout << "Toks mapping size: " << toks_cnt << std::endl;
    std::cout << "    Skipped toks: " << skipped_toks_cnt << std::endl;
    std::cout << "    Skipped toks cause no lemmas support: " << skipped_toks_no_lemmas_cnt << std::endl;
    // note: причин фильтрации по лемме может быть несколько; наиболее типичные: 1) лемма отсутствует в модели лемм (малочастотная), 2) токен слабо поддержан леммой (отсечение порогом представительности)
    std::string stnle_str;
    for (const auto w : skipped_toks_no_lemmas_examples)
      stnle_str += std::string(" ") + w;
    std::cout << "    Skipped toks cause no lemmas support examples:" << stnle_str << std::endl;
    std::cout << "Total LEX embeddings: " << (clear_embs_cnt + mixed_embs_cnt) << std::endl;
    std::cout << "    Clear embeddings: " << clear_embs_cnt << std::endl;
    std::cout << "    Mixed embeddings: " << mixed_embs_cnt << std::endl;
  }

  void rebalance(std::map<size_t, size_t>& lcmap)
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
