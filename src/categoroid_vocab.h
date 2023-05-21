#ifndef SRC_CATEGOROID_VOCAB_H_
#define SRC_CATEGOROID_VOCAB_H_

#include "str_conv.h"
#include "original_word2vec_vocabulary.h"


#include <string>
#include <vector>
#include <utility>
#include <set>
#include <fstream>
#include <iostream>
#include <regex>


// Словарь категориальных групп
class CategoroidsVocabulary
{
public:
  bool load(const std::string& filename, std::shared_ptr<OriginalWord2VecVocabulary> wordsVocabulary)
  {
    records.clear();
    if ( !wordsVocabulary )
      return false;
    std::ifstream ifs(filename);
    if (!ifs.good())
      return false;
    std::string line;
    while ( std::getline(ifs, line).good() )
    {
      StrConv::trim(line);
      if (line.empty()) continue;
      const std::regex space_re("\\s+");
      std::vector<std::string> items { std::sregex_token_iterator(line.cbegin(), line.cend(), space_re, -1), std::sregex_token_iterator() };
      if ( !check_items_helper(items, filename, line) )
        return false;
      const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
      size_t hyper_idx = wordsVocabulary->word_to_idx(items[0]);
      if (hyper_idx == INVALID_IDX)
      {
        std::cerr << "Skip record in " << filename << ": unknown hypernym word '" << items[0] << "'" << std::endl;
        continue;
      }
      for (size_t i = 1; i < items.size(); ++i)
      {
        size_t idx = wordsVocabulary->word_to_idx(items[i]);
        if (idx == INVALID_IDX)
        {
          std::cerr << "Skip unknown word '" << items[i] << "' in " << filename << std::endl;
          continue;
        }
        records.push_back( std::make_pair( hyper_idx, idx ) );
      }
    }
    print_stat_dbg();
    return true;
  }
  bool empty() const
  {
    return records.empty();
  }
  const std::pair<size_t,size_t>& get_random(unsigned long long next_random) const
  {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    return records[ next_random % records.size() ];
  }
  bool load_words_list(const std::string& filename)
  {
    words_list.clear();
    std::ifstream ifs(filename);
    if (!ifs.good())
      return false;
    std::string line;
    while ( std::getline(ifs, line).good() )
    {
      StrConv::trim(line);
      if (line.empty()) continue;
      const std::regex space_re("\\s+");
      std::vector<std::string> items { std::sregex_token_iterator(line.cbegin(), line.cend(), space_re, -1), std::sregex_token_iterator() };
      if ( !check_items_helper(items, filename, line) )
        return false;
      std::copy(items.begin(), items.end(), std::inserter(words_list, words_list.begin()));
    }
    return true;
  }
  bool in_words_list(const std::string& word) const
  {
    return words_list.find(word) != words_list.end();
  }
private:
  // вектор пар слов, связанных родовидовой связью
  std::vector< std::pair<size_t, size_t> > records;
  // множество слов, содержащихся в файле категороидов
  std::set<std::string> words_list;

  void print_stat_dbg() const
  {
    std::cout << "Categoroid pairs count = " << records.size() << std::endl;
    std::set<size_t> w;
    for (auto& r : records)
    {
      w.insert(r.first);
      w.insert(r.second);
    }
    std::cout << "Categoroid words count = " << w.size() << std::endl;
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
};



#endif /* SRC_CATEGOROID_VOCAB_H_ */
