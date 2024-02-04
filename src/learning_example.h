#ifndef LEARNING_EXAMPLE_H_
#define LEARNING_EXAMPLE_H_

#include <vector>
#include <utility>
#include <tuple>


// алгоритм стягивания/отталкивания (для работы со внешними словарями)
enum ExtVocabAlgo
{
  evaFirstWithOther,    // стягивание к первому слову словаря
  evaPairwise,          // попарное притяжение
  evaFirstWeighted      // стягиваие к первому, сила связи имеет вес
};


// структура для коррекции по внешнему словарю
struct ExtVocabExample
{
  size_t dims_from;
  size_t dims_to;
  size_t word1;
  size_t word2;
  float weight;
  ExtVocabAlgo algo;
  float e_dist_lim;
  ExtVocabExample(const std::pair<size_t, size_t>& d, const std::tuple<size_t, size_t, float>& w, const ExtVocabAlgo a, const float edl)
    : dims_from(d.first), dims_to(d.second), word1(std::get<0>(w)), word2(std::get<1>(w)), weight(std::get<2>(w)), algo(a), e_dist_lim(edl)
    { }
};

// структура, представляющая обучающий пример
struct LearningExample
{
  size_t word;                                              // индекс слова
  std::vector<size_t> dep_context;                          // индексы синтаксических контекстов
  std::vector<size_t> assoc_context;                        // индексы ассоциативных контекстов
  std::vector<ExtVocabExample> ext_vocab_data;              // дополнительные воздействия на основе данных внешних словарей
};


#endif /* LEARNING_EXAMPLE_H_ */
