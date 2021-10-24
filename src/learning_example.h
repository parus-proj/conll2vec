#ifndef LEARNING_EXAMPLE_H_
#define LEARNING_EXAMPLE_H_

#include <vector>
#include <utility>
#include <tuple>


// структура, представляющая обучающий пример
struct LearningExample
{
  size_t word;                                              // индекс слова
  std::vector<size_t> dep_context;                          // индексы синтаксических контекстов
  std::vector<size_t> assoc_context;                        // индексы ассоциативных контекстов
  std::vector<std::pair<size_t, size_t>> derivatives;       // индексы ассоциатов по деривации
  std::vector<std::tuple<size_t, size_t, float>> rassoc;    // индексы ассоциатов, согласно надежному источнику + степень их близости
  std::vector<std::pair<size_t, size_t>> categoroids;       // индексы категориально близких слов (согласно надежному источнику)
};


#endif /* LEARNING_EXAMPLE_H_ */
