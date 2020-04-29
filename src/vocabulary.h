#ifndef VOCABULARY_H_
#define VOCABULARY_H_

#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <list>
#include <iostream>


// данные словаря
struct VocabularyData
{
  // слово или контекст
  std::string word;
  // абсолютная частота
  uint64_t cn;
  // конструктор
  VocabularyData(const std::string& theWord, const uint64_t theFrequency)
  : word(theWord), cn(theFrequency)
  {}
};


// базовый класс словаря
class CustomVocabulary
{
public:
  // конструктор
  CustomVocabulary()
  {
  }
  // деструктор
  virtual ~CustomVocabulary()
  {
  }
  // получение индекса в словаре по тексту слова/контекста
  virtual size_t word_to_idx(const std::string& word) const = 0;
  // получение данных словаря по индексу
  inline const VocabularyData& idx_to_data(size_t word_idx) const
  {
    // без валидации word_idx для скорости
    return vocabulary[word_idx];
  }
  // получение размера словаря
  size_t size() const
  {
    return vocabulary.size();
  }
  // вычисление суммы абсолютных частот слов словаря
  uint64_t cn_sum() const
  {
    //return std::reduce(vocabulary.cbegin(), vocabulary.cend(), 0, [](const uint64_t& sum, const VocabularyData& r) -> uint64_t { return sum + r.cn; });
    //return std::accumulate(vocabulary.cbegin(), vocabulary.cend(), 0, [](const uint64_t& sum, const VocabularyData& r) -> uint64_t { return sum + r.cn; });
    return std::accumulate( vocabulary.cbegin(), vocabulary.cend(),
                            static_cast<uint64_t>(0),
                            [](const uint64_t& sum, const VocabularyData& r) -> uint64_t { return sum + r.cn; } );
  }
protected:
  std::vector<VocabularyData> vocabulary;
};


#endif /* VOCABULARY_H_ */
