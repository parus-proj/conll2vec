#ifndef RA_VOCAB_H_
#define RA_VOCAB_H_

#include "original_word2vec_vocabulary.h"


#include <string>
#include <vector>
#include <tuple>
#include <set>
#include <fstream>
#include <iostream>
#include <regex>


// Словарь надежных ассоциативных пар (для доп. тренировки ассоциативной модели)
class ReliableAssociativesVocabulary
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
      std::vector<std::string> rec_items { std::sregex_token_iterator(line.cbegin(), line.cend(), space_re, -1), std::sregex_token_iterator() };
      if (rec_items.size() != 3)
      {
        std::cerr << "Invalid record in " << filename << ": " << line << std::endl;
        return false;
      }
      const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
      size_t idx1 = wordsVocabulary->word_to_idx(rec_items[0]);
      size_t idx2 = wordsVocabulary->word_to_idx(rec_items[1]);
      if (idx1 == INVALID_IDX || idx2 == INVALID_IDX)
      {
        std::cerr << "Unknown word in '" << line << "' in " << filename << std::endl;
        continue;
      }
      float val = 0;
      try { val = std::stof(rec_items[2]); }
      catch (...) {
        std::cerr << "Invalid float value in '" << line << "' in " << filename << std::endl;
        continue;
      }
      records.push_back( std::make_tuple( idx1, idx2, val ) );
    }
    print_stat_dbg();
    return true;
  }
  const std::tuple<size_t, size_t, float>& get_random(unsigned long long next_random) const
  {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    return records[ next_random % records.size() ];
  }
private:
  // вектор пар слов, связанных "деривационно"
  std::vector< std::tuple<size_t, size_t, float> > records;

  void print_stat_dbg() const
  {
    std::cout << "Reliable assoc. pairs count = " << records.size() << std::endl;
    std::set<size_t> w;
    for (auto& r : records)
    {
      w.insert(std::get<0>(r));
      w.insert(std::get<1>(r));
    }
    std::cout << "RA words count = " << w.size() << std::endl;
  }
};



#endif /* RA_VOCAB_H_ */
