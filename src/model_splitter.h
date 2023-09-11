#ifndef SRC_MODEL_SPLITTER_H_
#define SRC_MODEL_SPLITTER_H_

#include "vectors_model.h"
#include "str_conv.h"


#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <algorithm>


// Логика разделения модели на подмодели псевдооснов, суффиксов и полных слов.
// (для более компактного представления в нейросетях и лучшего покрытия словаря)
/*
 * Выгоды подхода:
 * 1. Если у нас в модели есть только токены (отвинченный, отвинченную и отвинченного), а также суффиксы (-ому, -ым, -ых и др.), то
 *    выделение псевдоосновы отвинченн- позволит работать и с токенами, которых нет в модели (отвинченному, отвинченным и др.).
 * Риски подхода:
 * 1. Смешение неблизких значений. Пусть, например, в модели есть токены (блесток, блестка), (блестящее), (блестяще), а также (блестела, блестит, блестело, блестели).
 *    Предположим, на основе глагольной группы токенов мы выделим псевдооснову блест-. Для других же слов псевдооснова не выделится (не проходим критерии надежности).
 *    Тогда в модель попадет псевдооснова блест- (с глагольной категориальной семантикой), но она будет использоваться и при работе, например, с
 *    токенами "блестки", "белстящая" (их нет в словаре полных слов).
 * 2. Смешение может происходить по разным причинам. Например, (поезду, поездом, поезда) vs.(поездка, поездок, поездки). Если для обоих слов наберется достаточно
 *    токенов, то их отсеет фильтр дубликатов. Но если одна группа слов частотно не породит псевдооснову, но она породится от второй группы слов, то
 *    токены первой гурппы, не попавшие в модель, будут интерпретироваться как слова второй группы.
 * 3. Еще пример. (протектор, протектору, протектора) vs. (протекторат). Cуффикс -ат, определенно, есть. Поэтому протекторат категориально может превратиться в протектор.
 * 4. Регулярные случаи -- "купил" vs. "купился".
 *
 * TODO: АЛГОРИТМ ТРЕБУЕТ ПЕРЕСМОТРА, РИСКИ НЕДООЦЕНЕНЫ (смешение значений происходит чаще, чем предполагалось)!!!
 * TODO: ЭТА ЭВРИСТИКА МОЖЕТ РАБОТАТЬ ДЛЯ АССОЦИАТИВНОЙ ЧАСТИ, НЕ ДЛЯ КАТЕГОРИАЛЬНОЙ
*/


class ModelSplitter
{
public:
  static void run( const std::string& model_fn, const std::string& tlm_fn, const std::string& oov_fn, size_t size_g )
  {
    // 1. Загружаем модель
    std::cout << "Model loading..." << std::endl;
    VectorsModel vm;
    if ( !vm.load(model_fn) )
      return;

    // 2. Загружаем информацию о токенах (их отображение в леммы)
    std::cout << "Token-to-lemmas map loading..." << std::endl;
    std::map<std::string, std::map<size_t, size_t>> t2l_map;
    std::ifstream t2l_ifs( tlm_fn.c_str() );
    std::string buf;
    std::vector<std::string> parts;
    parts.reserve(100);
    while ( std::getline(t2l_ifs, buf).good() )
    {
      parts.clear();
      StrUtil::split_by_space(buf, parts);
      if ( parts.size() < 3 || parts.size() % 2 == 0 ) continue;    // skip invalid records
      const std::string token = parts[0];
      // парсим список лемм
      bool isParseOk = true;
      std::map<size_t, size_t> lcmap;
      for ( size_t i = 0; i < ((parts.size()-1)/2); ++i )
      {
        const auto& lemma = parts[i*2+1];
        const auto& cnt_str = parts[i*2+2];
        size_t lemma_idx = vm.get_word_idx_fast(lemma);
        if ( lemma_idx == vm.words_count ) continue;
        size_t cnt = 0;
        try { cnt = std::stoi(cnt_str); } catch (...) { isParseOk = false; break; }
        if (cnt < 50) continue; // отсечем шум
        lcmap[lemma_idx] = cnt;
      }
      if (!isParseOk || lcmap.empty()) continue;
      t2l_map[token] = lcmap;
    }

    // 3. Отберем токены, которым соответствует только одна лемма. Построим обратное отображение.
    std::cout << "Inverted map building..." << std::endl;
    std::map<size_t, std::vector<std::u32string>> l2t_map;  // отображение из lemma_id в список соответствующих токенов (которые точно относятся к этой лемме)
    for (auto& i : t2l_map)
    {
      if (i.second.size() == 1)
        l2t_map[i.second.begin()->first].push_back( StrConv::To_UTF32(i.first) );
    }

    // 4. Загрузим словарь суффиксов
    std::cout << "Sfx vocab loading..." << std::endl;
    std::set<std::u32string> sfxs;
    std::ifstream oov_ifs( oov_fn.c_str() );
    while ( std::getline(oov_ifs, buf).good() )
    {
      std::vector<std::string> parts;
      StrUtil::split_by_whitespaces(buf, parts);
      if ( parts.size() != 2 ) continue;    // skip invalid records
      sfxs.insert(StrConv::To_UTF32(parts[0]).substr(5));
    }

    // 5. Пытаемся разделить словоформы каждой леммы на основу и суффикс.
    //    Если количество словоформ достаточное (для надежного решения) и все они делятся, то делим.
    std::cout << "Pseudo-inflectional-paradigm search..." << std::endl;
    std::map<std::u32string, size_t> stems2lemma;
    std::set<std::u32string> invalid_common_prefixes;
    for (auto& i : l2t_map)
    {
//      const bool DEBUG_MARKER = ( vm.vocab[i.first].find("блест") == 0 );
      const auto& toks = i.second;
//      if (DEBUG_MARKER) {
//        std::cout << "see " << vm.vocab[i.first] << std::endl;
//        std::string l = "";
//        for(auto& j : toks)
//          l += " " + StrConv::To_UTF8(j);
//        std::cout << "    " << l << std::endl;
//      }
//      if (DEBUG_MARKER && toks.size() < 3) { std::cout << "SKIP: toks count" << std::endl; }
      if (toks.size() < 3) continue;
      std::u32string common_prefix = toks.front();
      for (size_t j = 1; j < toks.size(); ++j)
      {
        size_t k = 0;
        while ( k < common_prefix.length() && k < toks[j].length() && common_prefix[k] == toks[j][k])
          ++k;
        common_prefix.erase(k);
      }
//      if (DEBUG_MARKER && common_prefix.empty()) { std::cout << "SKIP: no common prefix" << std::endl; }
      if (common_prefix.empty()) continue;
      // Сочетания с другими суффиксами могут образовывать слова от другой леммы: банк_ + _OOV_рот, банк_ + _OOV_ир, протектор_ + _OOV_ат, рекорд_ + _OOV_смен.
      //   Это относительно безопасно, т.к. стратегия токенизации предполагает отрезание минимального суффикса.
      //   Но _протекторат_ может не попасть в списки основа+суффикс и тогда может возникать путаница.
      //   todo: подумать, насчет такой фильтрации
      //   Вероятно, имеет смысл ограничить и минимальную длину основы.
//      if (DEBUG_MARKER && common_prefix.length() < 5) { std::cout << "SKIP: short common prefix" << std::endl; }
      if (common_prefix.length() < 5) continue;
//      std::cout << "     " << StrConv::To_UTF8(common_prefix) << std::endl;
      bool all_found = true;
      for (size_t j = 0; j < toks.size(); ++j)
      {
        auto token_suffix = toks[j].substr(common_prefix.length());
        if (token_suffix.empty()) continue;
        if ( sfxs.find(token_suffix) == sfxs.end() )
        {
//          if (DEBUG_MARKER)
//            std::cout << "     sfx not found: " << StrConv::To_UTF8(token_suffix) << std::endl;
          all_found = false;
          break;
        }
      }
//      if (DEBUG_MARKER && !all_found) { std::cout << "SKIP: not all found" << std::endl; }
      if (!all_found) continue;
//      std::cout << "     all sfx found" << std::endl;
      // Смотрим, не породились ли дубликаты псевдооснов (их надо исключить)
      // долг - долгий, способ - способный (способен), обратиться - обратить, признаться - признать, поезд - поездка (поездок), подход - подходить и т.п.
      if ( invalid_common_prefixes.find(common_prefix) != invalid_common_prefixes.end() )
      {
//        if (DEBUG_MARKER) { std::cout << "SKIP: doubler" << std::endl; }
//        std::cout << "     dbl stem: " << StrConv::To_UTF8(common_prefix) << std::endl;
        continue;
      }
      auto dIt = stems2lemma.find(common_prefix);
      if (dIt != stems2lemma.end())
      {
//        std::cout << "     dbl stem: " << StrConv::To_UTF8(common_prefix) << std::endl;
        invalid_common_prefixes.insert(common_prefix);
        stems2lemma.erase(dIt);
//        if (DEBUG_MARKER) { std::cout << "SKIP: doubler" << std::endl; }
        continue;
      }
      stems2lemma[common_prefix] = i.first;
//      std::cout << "     stem accepted: " << StrConv::To_UTF8(common_prefix) << std::endl;
    }

    // 6. Создаем список токенов, исключаемых из модели полных форм
    std::set<size_t> excl_toks;
    for (auto& i : stems2lemma)
    {
      auto& tlist = l2t_map[i.second];
      for (auto& j : tlist)
        excl_toks.insert( vm.get_word_idx_fast( StrConv::To_UTF8(j) ) );
    }

    std::cout << "Saving..." << std::endl;
    // 7. Сохраняем вектора псевдооснов
    size_t stem_size = vm.emb_size - size_g;
    std::string stems_model_fn = model_fn + ".stems";
    FILE *stems_fo = fopen(stems_model_fn.c_str(), "wb");
    fprintf(stems_fo, "%lu %lu\n", stems2lemma.size()+sfxs.size(), stem_size);
    for (auto& i : stems2lemma)
    {
      VectorsModel::write_embedding_slice(stems_fo, StrConv::To_UTF8(i.first), &vm.embeddings[i.second * vm.emb_size], 0, stem_size);
    }
    std::set<size_t> sfx_toks;
    for ( auto& i : sfxs)
    {
      auto oovstr = std::string("_OOV_")+StrConv::To_UTF8(i);
      auto oovidx = vm.get_word_idx( oovstr );
      VectorsModel::write_embedding_slice(stems_fo, oovstr, &vm.embeddings[oovidx * vm.emb_size], 0, stem_size);
      sfx_toks.insert(oovidx);
      excl_toks.insert(oovidx);
    }
    fclose(stems_fo);

    // 8. Сохраняем вектора суффиксов
    std::string sfx_model_fn = model_fn + ".sfx";
    FILE *sfx_fo = fopen(sfx_model_fn.c_str(), "wb");
    fprintf(sfx_fo, "%lu %lu\n", sfx_toks.size(), size_g);
    for (auto& i : sfx_toks)
    {
      VectorsModel::write_embedding_slice(sfx_fo, vm.vocab[i], &vm.embeddings[i * vm.emb_size], stem_size, vm.emb_size);
    }
    fclose(sfx_fo);

    // 9. Сохраняем вектора оставшихся полных слов
    std::string forms_model_fn = model_fn + ".forms";
    FILE *frm_fo = fopen(forms_model_fn.c_str(), "wb");
    fprintf(frm_fo, "%lu %lu\n", vm.vocab.size()-excl_toks.size(), vm.emb_size);
    for (size_t a = 0; a < vm.vocab.size(); ++a)
    {
      if ( excl_toks.find(a) != excl_toks.end() ) continue;
      VectorsModel::write_embedding(frm_fo, vm.vocab[a], &vm.embeddings[a * vm.emb_size], vm.emb_size);
    }
    fclose(frm_fo);


  } // method-end: run(...)
}; // class-decl-end: ModelSplitter



#endif /* SRC_MODEL_SPLITTER_H_ */
