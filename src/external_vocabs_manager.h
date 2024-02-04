#ifndef SRC_EXT_VOCABS_MANAGER_H_
#define SRC_EXT_VOCABS_MANAGER_H_

#include "str_conv.h"
#include "learning_example.h"


#include <string>
#include <vector>
#include <utility>
#include <set>
#include <fstream>
#include <iostream>
#include <atomic>


// Информация о словаре и о том, как его использовать для обучения векторной модели (а также сами словарные данные)
struct VocabUsageInfo
{
  std::pair<size_t, size_t> dims_range;     // измерения векторной модели, к которым применяется алгортим дополнительного стягивания/отталкивания на основе словарных данных
  std::pair<float, float> fraction_range;   // на какой стадии обучения использовать (проценты от всей длительности обучения)
  std::string vocab_filename;               // имя файла с данными словаря
  ExtVocabAlgo algo;                        // алгоритм стягивания/отталкивания
  size_t rate = 0;  // как часто использовать данные словаря при обучении (с каждым rate обучающим примером)
  size_t pack = 0;  // по сколько словарных данных подавать в один эпизод обучения
  float e_dist_limit = 0; // предел стягивания (по евклидову расстоянию)

  std::vector< std::tuple<size_t, size_t, float> > data;  // сами данные словаря -- пары индексов слов и вес связи
  mutable std::atomic_uint counter{0};                    // счетчик для выбора обучающих примеров с заданной частотой сэмплирования (rate)
};


// Менеджер внешних словарей, содержащих информацию о связанности слов
class ExternalVocabsManager
{
public:
  // загрузка информации о словарях
  bool load(const std::string& filename)
  {
    records.clear();
    std::ifstream ifs(filename);
    if (!ifs.good())
      return false;
    std::string line;
    while ( std::getline(ifs, line).good() )
    {
      StrConv::trim(line);
      if (line.empty()) continue;
      if (line[0] == '#') continue;

      std::vector<std::string> rec_fields;
      StrUtil::split_by_whitespaces(line, rec_fields);

      if (rec_fields.size() != 9)
      {
        std::cerr << "Skip invalid record: " << line << " (" << filename << ")" << std::endl;
        continue;
      }

      try
      {
        std::unique_ptr<VocabUsageInfo> rptr = std::make_unique<VocabUsageInfo>();
        auto& r = *rptr;
        r.dims_range = std::make_pair( std::stoi(rec_fields[0]), std::stoi(rec_fields[1]) );
        r.fraction_range = std::make_pair( std::stof(rec_fields[2])/100.0, std::stof(rec_fields[3])/100.0 );
        r.vocab_filename = rec_fields[4];
        r.algo = static_cast<ExtVocabAlgo>( std::stoi(rec_fields[5]) );
        r.rate = std::stoi(rec_fields[6]);
        r.pack = std::stoi(rec_fields[7]);
        r.e_dist_limit = std::stof(rec_fields[8]);
        records.push_back( std::move(rptr) );
      }
      catch (...) {
        std::cerr << "Skip invalid record: " << line << " (" << filename << ")" << std::endl;
        continue;
      }
    }
    print_table_dbg();
    return true;
  }
  // загрузка словарей
  bool load_vocabs(std::shared_ptr<OriginalWord2VecVocabulary> wordsVocabulary)
  {
    if ( !wordsVocabulary )
      return false;
    for (auto& vptr : records)
    {
      auto& v = *vptr;
      v.data.clear();
      v.counter = 0;
      std::ifstream ifs(v.vocab_filename);
      if (!ifs.good())
      {
        std::cerr << "File not found: " << v.vocab_filename << std::endl;
        return false;
      }
      std::string line;
      while ( std::getline(ifs, line).good() )
      {
        StrConv::trim(line);
        if (line.empty()) continue;
        std::vector<std::string> items;
        StrUtil::split_by_whitespaces(line, items);
        if ( !check_items_helper(items, v.vocab_filename, line) )
          return false;
        switch ( v.algo )
        {
        case evaFirstWithOther: first_with_other_helper(v, items, wordsVocabulary, v.vocab_filename); break;
        case evaPairwise: pairwise_helper(v, items, wordsVocabulary, v.vocab_filename); break;
        case evaFirstWeighted: first_weighted_helper(v, items, wordsVocabulary, v.vocab_filename); break;
        }
      }
      print_stat_dbg(v);
    } // for all vocabs
    return true;
  }
  // наполенние структуры result обучающими данными (вызывается из нескольких потоков!!!)
  void get(std::vector<ExtVocabExample>& result, const float fraction, unsigned long long next_random) const
  {
    for (auto& vptr : records)
    {
      auto& v = *vptr;
      if ( v.data.empty() )
        continue;
      if ( fraction < v.fraction_range.first || fraction > v.fraction_range.second )
        continue;
      if ( ++v.counter % v.rate == 0 )
      {
        for (size_t i = 0; i < v.pack; ++i)
        {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          auto& selected = v.data[ next_random % v.data.size() ];
          result.emplace_back( v.dims_range, selected, v.algo, v.e_dist_limit );
        }
      }
    }
  }
private:
  // вектор информации о словарях
  std::vector< std::unique_ptr<VocabUsageInfo> > records;

  void print_table_dbg() const
  {
    std::cout << "External vocabs table" << std::endl;
    std::cout << "DIM_FROM   DIM_TO   FRAC_FROM   FRAC_TO   FILE   ALGO   RATE   PACK   EDIST_LIM" << std::endl;
    for (auto& rptr : records)
    {
      auto& r = *rptr;
      std::cout << r.dims_range.first << "  " << r.dims_range.second << "  "
                << r.fraction_range.first << "  " << r.fraction_range.second << "  "
                << r.vocab_filename << "  " << ((size_t)r.algo) << "  "
                << r.rate << "  " << r.pack << "  " << r.e_dist_limit << std::endl;
    }
  }

  bool check_items_helper(const std::vector<std::string>& items, const std::string& filename, const std::string& line)
  {
    if (items.size() < 2)
    {
      std::cerr << "Invalid record (count) in " << filename << ": " << line << std::endl;
      return false;
    }
    std::set<std::string> dedoubler;
    std::copy(items.begin(), items.end(), std::inserter(dedoubler, dedoubler.begin()));
    if (items.size() != dedoubler.size())
    {
      std::cerr << "Invalid record (duplicates) in " << filename << ": " << line << std::endl;
      return false;
    }
    return true;
  }

  void first_with_other_helper(VocabUsageInfo& v, const std::vector<std::string>& items, std::shared_ptr<OriginalWord2VecVocabulary> wordsVocabulary, const std::string& filename)
  {
    constexpr size_t INVALID_IDX = std::numeric_limits<size_t>::max();
    size_t hyper_idx = wordsVocabulary->word_to_idx(items[0]);
    if (hyper_idx == INVALID_IDX)
    {
      //std::cerr << "Skip record in " << filename << ": unknown hypernym word '" << items[0] << "'" << std::endl;
      return;
    }
    for (size_t i = 1; i < items.size(); ++i)
    {
      size_t idx = wordsVocabulary->word_to_idx(items[i]);
      if (idx == INVALID_IDX)
      {
        //std::cerr << "Skip unknown word '" << items[i] << "' in " << filename << std::endl;
        continue;
      }
      v.data.push_back( std::make_tuple( hyper_idx, idx, 1.0 ) );
    }
  }

  void pairwise_helper(VocabUsageInfo& v, const std::vector<std::string>& items, std::shared_ptr<OriginalWord2VecVocabulary> wordsVocabulary, const std::string& filename)
  {
    constexpr size_t INVALID_IDX = std::numeric_limits<size_t>::max();
    std::vector<size_t> nest;
    for (auto& word : items)
    {
      size_t idx = wordsVocabulary->word_to_idx(word);
      if (idx == INVALID_IDX)
      {
        //std::cerr << "Skip unknown word '" << word << "' in " << filename << std::endl;
        continue;
      }
      nest.push_back(idx);
    }
    if (nest.size() < 2)
    {
      //std::cerr << "Too small nest in " << filename << ": " << items[0] << " ..." << std::endl;
      return;
    }
    for (size_t i = 0; i < nest.size()-1; ++i)
      for (size_t j = i+1; j < nest.size(); ++j)
        v.data.push_back( std::make_tuple( nest[i], nest[j], 1.0 ) );
  }

  void first_weighted_helper(VocabUsageInfo& v, const std::vector<std::string>& items, std::shared_ptr<OriginalWord2VecVocabulary> wordsVocabulary, const std::string& filename)
  {
    constexpr size_t INVALID_IDX = std::numeric_limits<size_t>::max();
    const size_t idx_1 = wordsVocabulary->word_to_idx(items[0]);
    if (idx_1 == INVALID_IDX)
    {
      //std::cerr << "Skip record in " << filename << ": unknown word '" << items[0] << "'" << std::endl;
      return;
    }
    const size_t idx_2 = wordsVocabulary->word_to_idx(items[1]);
    if (idx_2 == INVALID_IDX)
    {
      //std::cerr << "Skip record in " << filename << ": unknown word '" << items[1] << "'" << std::endl;
      return;
    }
    float val = 0;
    try { val = std::stof(items[2]); }
    catch (...) {
      //std::cerr << "Skip record in " << filename << ": invalid float value '" << items[2] << "'" << std::endl;
      return;
    }
    v.data.push_back( std::make_tuple( idx_1, idx_2, val ) );
  }

  void print_stat_dbg(VocabUsageInfo& v) const
  {
    std::cout << "Vocab: " << v.vocab_filename << std::endl;
    std::cout << "  pairs count = " << v.data.size() << std::endl;
    std::set<size_t> w;
    for (auto& r : v.data)
    {
      w.insert(std::get<0>(r));
      w.insert(std::get<1>(r));
    }
    std::cout << "  words count = " << w.size() << std::endl;
  }

};



#endif /* SRC_EXT_VOCABS_MANAGER_H_ */
