#ifndef LEARNING_EXAMPLE_H_
#define LEARNING_EXAMPLE_H_

#include <vector>
#include <utility>
#include <tuple>
#include <optional>


// структура, представляющая обучающий пример
struct LearningExample
{
  size_t word;                                              // индекс слова
  std::vector<size_t> dep_context;                          // индексы синтаксических контекстов
  std::vector<size_t> assoc_context;                        // индексы ассоциативных контекстов
  std::optional<std::pair<size_t, size_t>> derivatives;     // индексы ассоциатов по деривации
  std::optional<std::tuple<size_t, size_t, float>> rassoc;  // индексы ассоциатов, согласно надежному источнику + степень их близости
};


#endif /* LEARNING_EXAMPLE_H_ */
