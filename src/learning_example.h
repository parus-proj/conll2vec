#ifndef LEARNING_EXAMPLE_H_
#define LEARNING_EXAMPLE_H_

#include <vector>
#include <utility>
#include <tuple>


// алгоритм стягивания/отталкивания (для работы со внешними словарями)
enum ExtVocabAlgo
{
  evaFirstWithOther,    // стягивание к первому слову словаря
  evaPairwise           // попарное притяжение
};


// структура для коррекции по внешнему словарю
struct ExtVocabExample
{
  size_t dims_from;
  size_t dims_to;
  size_t word1;
  size_t word2;
  ExtVocabAlgo algo;
  ExtVocabExample(const std::pair<size_t, size_t>& d, const std::pair<size_t, size_t>& w, ExtVocabAlgo a)
    : dims_from(d.first), dims_to(d.second), word1(w.first), word2(w.second), algo(a)
    { }
};

// структура, представляющая обучающий пример
struct LearningExample
{
  size_t word;                                              // индекс слова
  std::vector<size_t> dep_context;                          // индексы синтаксических контекстов
  std::vector<size_t> assoc_context;                        // индексы ассоциативных контекстов
  std::vector<std::tuple<size_t, size_t, float>> rassoc;    // индексы ассоциатов, согласно надежному источнику + степень их близости
  std::vector<ExtVocabExample> ext_vocab_data;
};


#endif /* LEARNING_EXAMPLE_H_ */
