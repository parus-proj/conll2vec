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
class ModelSplitter
{
public:
  static void run( const std::string& model_fn, const std::string& tlm_fn, const std::string& oov_fn, size_t size_g, bool useTxtFmt = false )
  {
    // 1. Загружаем модель
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
      StrUtil::split_by_space(buf, parts);
      if ( parts.size() < 3 || parts.size() % 2 == 0 ) continue;    // skip invalid records
      std::string token = parts[0];
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
        if (cnt < 10) continue; // отсечем шум
        lcmap[lemma_idx] = cnt;
      }
      if (!isParseOk || lcmap.empty()) continue;
      t2l_map[token] = lcmap;
    }

    // 3. Отберем токены, которым соответствует только одна лемма. Построим обратное отображение.
    std::map<size_t, std::vector<std::u32string>> l2t_map;
    for (auto& i : t2l_map)
    {
      if (i.second.size() == 1)
        l2t_map[i.second.begin()->first].push_back( StrConv::To_UTF32(i.first) );
    }
    // добавление самих лемм (леммы может не быть в списке токенов по частотной причине, хотя она при этом присутствует в модели)
    for (auto& i : l2t_map)
    {
      auto lemma_str = StrConv::To_UTF32(vm.vocab[i.first]);
      auto it = std::find(i.second.cbegin(), i.second.cend(), lemma_str);
      if (it == i.second.end())
        i.second.push_back(lemma_str);
    }


    // 4. Загрузим словарь суффиксов
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
    std::map<std::u32string, size_t> stems2lemma;
    for (auto& i : l2t_map)
    {
      const auto& toks = i.second;
      if (toks.size() < 3) continue;
//      std::cout << "see " << vm.vocab[i.first] << std::endl;
//      std::string l = "";
//      for(auto& j : toks)
//        l += " " + StrConv::To_UTF8(j);
//      std::cout << "    " << l << std::endl;
      std::u32string common_prefix = toks.front();
      for (size_t j = 1; j < toks.size(); ++j)
      {
        size_t k = 0;
        while ( k < common_prefix.length() && k < toks[j].length() && common_prefix[k] == toks[j][k])
          ++k;
        common_prefix.erase(k);
      }
      if (common_prefix.empty()) continue;
      // Сочетания с другими суффиксами могут образовывать слова от другой леммы: банк_ + _OOV_рот, банк_ + _OOV_ир, протектор_ + _OOV_ат, рекорд_ + _OOV_смен.
      //   Это относительно безопасно, т.к. стратегия токенизации предполагает отрезание минимального суффикса.
      //   Но _протекторат_ может не попасть в списки основа+суффикс и тогда может возникать путаница.
      //   todo: подумать, насчет такой фильтрации
      //   Вероятно, имеет смысл ограничить и минимальную длину основы.
      if (common_prefix.length() < 5) continue;
//      std::cout << "     " << StrConv::To_UTF8(common_prefix) << std::endl;
      bool all_found = true;
      for (size_t j = 1; j < toks.size(); ++j)
      {
        auto token_suffix = toks[j].substr(common_prefix.length());
        if (token_suffix.empty()) continue;
        if ( sfxs.find(token_suffix) == sfxs.end() )
        {
//          std::cout << "     sfx not found: " << StrConv::To_UTF8(token_suffix) << std::endl;
          all_found = false;
          break;
        }
      }
      if (!all_found) continue;
//      std::cout << "     all sfx found" << std::endl;
      // Смотрим, не породились ли дубликаты псевдооснов (их надо исключить)
      // долг - долгий, способ - способный (способен), обратиться - обратить, признаться - признать, поезд - поездка (поездок), подход - подходить и т.п.
      auto dIt = stems2lemma.find(common_prefix);
      if (dIt != stems2lemma.end())
      {
//        std::cout << "     dbl stem: " << StrConv::To_UTF8(common_prefix) << std::endl;
        stems2lemma.erase(dIt);
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
        excl_toks.insert( vm.get_word_idx( StrConv::To_UTF8(j) ) );
    }

    // 7. Сохраняем вектора псевдооснов
    size_t stem_size = vm.emb_size - size_g;
    std::string stems_model_fn = model_fn + ".stems";
    FILE *stems_fo = fopen(stems_model_fn.c_str(), "wb");
    fprintf(stems_fo, "%lu %lu\n", stems2lemma.size(), stem_size);
    for (auto& i : stems2lemma)
    {
      VectorsModel::write_embedding_slice(stems_fo, useTxtFmt, StrConv::To_UTF8(i.first), &vm.embeddings[i.second * vm.emb_size], 0, stem_size);
    }
    fclose(stems_fo);

    // 8. Сохраняем вектора суффиксов
    std::set<size_t> sfx_toks;
    for (size_t a = 0; a < vm.vocab.size(); ++a)
    {
      if (vm.vocab[a].find("_OOV_") == 0)
        sfx_toks.insert(a);
    }
    std::string sfx_model_fn = model_fn + ".sfx";
    FILE *sfx_fo = fopen(sfx_model_fn.c_str(), "wb");
    fprintf(sfx_fo, "%lu %lu\n", sfx_toks.size(), size_g);
    for (auto& i : sfx_toks)
    {
      VectorsModel::write_embedding_slice(sfx_fo, useTxtFmt, vm.vocab[i], &vm.embeddings[i * vm.emb_size], stem_size, vm.emb_size);
    }
    fclose(sfx_fo);

    // 9. Сохраняем вектора оставшихся полных слов
    std::string forms_model_fn = model_fn + ".forms";
    FILE *frm_fo = fopen(forms_model_fn.c_str(), "wb");
    fprintf(frm_fo, "%lu %lu\n", vm.vocab.size()-excl_toks.size()-sfx_toks.size(), vm.emb_size);
    for (size_t a = 0; a < vm.vocab.size(); ++a)
    {
      if ( excl_toks.find(a) != excl_toks.end() || sfx_toks.find(a) != sfx_toks.end() ) continue;
      VectorsModel::write_embedding(frm_fo, useTxtFmt, vm.vocab[a], &vm.embeddings[a * vm.emb_size], vm.emb_size);
    }
    fclose(frm_fo);


  } // method-end: run(...)
}; // class-decl-end: ModelSplitter



#endif /* SRC_MODEL_SPLITTER_H_ */
