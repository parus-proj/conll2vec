#ifndef SRC_DERIVE_MAKER_H_
#define SRC_DERIVE_MAKER_H_

#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <iostream>

#include "str_conv.h"


// Построитель словаря деривативных гнезд по паттернам
class DerivativeNestsMaker
{
public:
  static void run( const std::string& main_vocab_fn, const std::string& patterns_fn, const std::string& result_fn )
  {
    std::vector<PatternData> patterns;
    if ( !load_patterns(patterns_fn, patterns) )
      return;
    std::set<std::u32string> words;
    if ( !load_main_vocab(main_vocab_fn, words) )
      return;

    std::vector< std::set<std::u32string> > nests;
    std::map<std::u32string, size_t> word2nestidx;

    for (auto w : words)
      for (auto p : patterns)
      {
        if ( !has_suffix(w, p.suffix, p.min_base) ) continue;
        auto new_word = w.substr( 0, w.length() - p.suffix.length() );
        new_word += p.replacement;
        if ( words.find(new_word) == words.end() ) continue;

        auto it1 = word2nestidx.find(w), it2 = word2nestidx.find(new_word);
        if ( it1 != word2nestidx.end() && it2 != word2nestidx.end() ) // случай, когда оба слова уже есть
        {
          if (it1->second == it2->second) // оба уже в одном гнезде -- ничего делать не надо
            continue;
          else
          {
            // требуется объединение гнезд
            auto& to_move = nests[it2->second];
            for (auto i : to_move)
              word2nestidx[i] = it1->second;
            to_move.clear();
          }
        }
        else
        {
          if ( it1 != word2nestidx.end() && it2 == word2nestidx.end() )
          {
            nests[it1->second].insert(new_word);
            word2nestidx[new_word] = it1->second;
          }
          else if ( it1 == word2nestidx.end() && it2 != word2nestidx.end() )
          {
            nests[it2->second].insert(w);
            word2nestidx[w] = it2->second;
          }
          else
          {
            size_t idx = nests.size();
            nests.push_back( std::set<std::u32string>() );
            nests[idx].insert(w);
            nests[idx].insert(new_word);
            word2nestidx[w] = idx;
            word2nestidx[new_word] = idx;
          }
        }
      } // for

    std::ofstream ofs(result_fn);
    if (!ofs.good())
    {
      std::cerr << "Can't open the file: " << result_fn << std::endl;
      return;
    }

    for (auto& n : nests)
    {
      if (n.empty()) continue;
      std::u32string nest;
      for (auto& w : n)
        nest += (nest.empty()) ? w : U" "+w;
      ofs << StrConv::To_UTF8(nest) << std::endl;
    }

  } // method-end
private:
  // хранилище паттернов
  struct PatternData
  {
    // минимально необходимая основа для применения паттерна
    size_t min_base;
    // суффикс
    std::u32string suffix;
    // замена суффиксу для порождения нового слова из деривативного гнезда
    std::u32string replacement;
    PatternData(size_t b, const std::u32string& s, const std::u32string& r): min_base(b), suffix(s), replacement(r) {}
  };

  static bool load_patterns(const std::string& patterns_fn, std::vector<PatternData>& patterns)
  {
    patterns.clear();
    std::ifstream ifs(patterns_fn);
    if (!ifs.good())
    {
      std::cerr << "Can't open the file: " << patterns_fn << std::endl;
      return false;
    }
    std::string line;
    while ( std::getline(ifs, line).good() )
    {
      if (line.empty())
        continue;
      if (line[0] == '#')
        continue;
      std::u32string line32 = StrConv::To_UTF32(line);
      std::vector<std::u32string> record_items;
      size_t tab_pos, pre_tab_pos = 0;
      while ( (tab_pos = line32.find(U"\t", pre_tab_pos)) != std::u32string::npos )
      {
        record_items.emplace_back( line32.substr(pre_tab_pos, tab_pos-pre_tab_pos) );
        pre_tab_pos = tab_pos+1;
      }
      record_items.emplace_back( line32.substr(pre_tab_pos) );
      if (record_items.size() == 1)
      {
        std::cerr << "Tab not found: " << line << std::endl;
        continue;
      }
      size_t base_size = (record_items.size() > 2) ? std::stoi(StrConv::To_UTF8(record_items[2])) : 5;
      patterns.emplace_back( base_size, record_items[0], record_items[1] );

    }
    return true;
  }

  static bool load_main_vocab(const std::string& main_vocab_fn, std::set<std::u32string>& words)
  {
    words.clear();
    std::ifstream ifs(main_vocab_fn);
    if (!ifs.good())
    {
      std::cerr << "Can't open the file: " << main_vocab_fn << std::endl;
      return false;
    }
    std::string line;
    while ( std::getline(ifs, line).good() )
    {
      if (line.empty())
        continue;
      size_t space_pos = line.find(" ");
      if (space_pos == std::string::npos)
      {
        std::cerr << "Tab not found: " << line << std::endl;
        continue;
      }
      auto&& word = StrConv::To_UTF32( line.substr(0, space_pos) );
      if (word.length() > 4)  // короткие слова даже не загружаем
        words.insert( word );
    }
    return true;
  }

  static bool has_suffix(const std::u32string& word, const std::u32string& suffix, size_t min_base)
  {
    if ( word.length() < (suffix.length() + min_base) )
      return false;
    std::u32string::const_reverse_iterator wIt = word.rbegin(), sIt = suffix.rbegin(), sItEnd = suffix.rend();
    while ( sIt != sItEnd )
    {
      if (*wIt != *sIt)
        return false;
      ++sIt;
      ++wIt;
    }
    return true;
  }

}; // class-decl-end


#endif /* SRC_DERIVE_MAKER_H_ */
