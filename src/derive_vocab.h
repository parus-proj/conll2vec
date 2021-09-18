#include "original_word2vec_vocabulary.h"


#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <iostream>
#include <regex>


#ifndef SRC_DERIVE_VOCAB_H_
#define SRC_DERIVE_VOCAB_H_

// Словарь деривативных гнезд (для тренировки деривативной ассоциации)
class DerivativeVocabulary
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
      const std::regex space_re("\\s+");
      std::vector<std::string> nest_items { std::sregex_token_iterator(line.cbegin(), line.cend(), space_re, -1), std::sregex_token_iterator() };
      if (nest_items.size() < 2)
      {
        std::cerr << "Invalid record in " << filename << ": " << line << std::endl;
        return false;
      }
      std::set<std::string> dedoubler;
      std::copy(nest_items.begin(), nest_items.end(), std::inserter(dedoubler, dedoubler.begin()));
      if (nest_items.size() != dedoubler.size())
      {
        std::cerr << "Invalid record in " << filename << ": " << line << std::endl;
        return false;
      }
      std::vector<size_t> nest;
      const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
      for (auto& word : nest_items)
      {
        size_t idx = wordsVocabulary->word_to_idx(word);
        if (idx == INVALID_IDX)
        {
          std::cerr << "Unknown word '" << word << "' in " << filename << std::endl;
          continue;
        }
        nest.push_back(idx);
      }
      if (nest.size() < 2)
      {
        std::cerr << "Too small nest in " << filename << ": " << line << std::endl;
        continue;
      }
      for (size_t i = 0; i < nest.size()-1; ++i)
        for (size_t j = i+1; j < nest.size(); ++j)
          records.push_back( std::vector<size_t>( {nest[i], nest[j]} ) );
    }
    print_stat_dbg();
    return true;
  }
  const std::vector<size_t>& get_random(unsigned long long next_random) const
  {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    return records[ next_random % records.size() ];
  }
private:
  // вектор пар слов, связанных "деривационно"
  std::vector< std::vector<size_t> > records;

  void print_stat_dbg() const
  {
    std::cout << "Deriv. pairs count = " << records.size() << std::endl;
    std::set<size_t> w;
    for (auto& r : records)
    {
      w.insert(r[0]);
      w.insert(r[1]);
    }
    std::cout << "Deriv. words count = " << w.size() << std::endl;
  }
};



#endif /* SRC_DERIVE_VOCAB_H_ */
