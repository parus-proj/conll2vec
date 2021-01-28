#ifndef VOCABULARY_H_
#define VOCABULARY_H_

#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <list>
#include <iostream>
#include <cmath>


// данные словаря
struct VocabularyData
{
  // слово или контекст
  std::string word;
  // абсолютная частота
  uint64_t cn;
  // вероятность сэмплирования (в соответствии с коэффициентом subsampling)
  float sample_probability;
  // конструктор
  VocabularyData(const std::string& theWord, const uint64_t theFrequency)
  : word(theWord), cn(theFrequency), sample_probability(1.0)
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
  // вычисление вероятностей сэмплирования для заданного коэффициента
  void sampling_estimation(float sample)
  {
    if (sample == 0)
      return;
    auto total = cn_sum();
    float wc_mul_sample = total * sample;
    for (auto& r : vocabulary)
    {
      float t_to_f = wc_mul_sample / r.cn;
      float prob = t_to_f + std::sqrt(t_to_f);
      r.sample_probability = (prob > 1) ? 1.0 : prob;
    }
  }
  // добавление записи в словарь
  virtual void append(const std::string& word, uint64_t cn)
  {
    vocabulary.emplace_back(word, cn);
  }
  // приписывание заданного суффикса всем словам словаря
  void suffixize(const std::string& suffix)
  {
    for (auto& r : vocabulary)
      r.word.append(suffix);
  }
protected:
  std::vector<VocabularyData> vocabulary;
};


#endif /* VOCABULARY_H_ */
