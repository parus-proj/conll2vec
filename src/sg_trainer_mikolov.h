#ifndef SRC_SG_TRAINER_MIKOLOV_H_
#define SRC_SG_TRAINER_MIKOLOV_H_

#include <algorithm>
#include <functional>
#include <numeric>
//#include <execution>

#include "trainer.h"


class SgTrainer_Mikolov : public CustomTrainer
{
public:
  // конструктор
  SgTrainer_Mikolov( std::shared_ptr< CustomLearningExampleProvider> learning_example_provider,
                     std::shared_ptr< CustomVocabulary > words_vocabulary,
                     std::shared_ptr< CustomVocabulary > contexts_vocabulary,
                     size_t embedding_size = 100,
                     size_t epochs = 5,
                     float learning_rate = 0.025,
                     size_t negative_count = 5 )
  : CustomTrainer(learning_example_provider, words_vocabulary, contexts_vocabulary, words_vocabulary, contexts_vocabulary, embedding_size, epochs, learning_rate, negative_count)
  {
    InitUnigramTable_w();
  }
  // деструктор
  virtual ~SgTrainer_Mikolov()
  {
  }
  // функция, реализующая модель обучения skip-gram
  void learning_model(const LearningExample& le, float *neu1, float *neu1e)
  {
    if (le.context.size() == 0) return;
    // цикл по контекстам
    for (auto&& ctx_idx : le.context)
    {
      // зануляем текущие значения ошибок (это частная производная ошибки E по выходу скрытого слоя h)
      std::fill(neu1e, neu1e+layer1_size, 0.0);
      // вычисляем смещение вектора, соответствующего очередному контексту
      float *ctxVectorPtr = syn0 + ctx_idx * layer1_size;
      size_t target;
      int label; // знаковое целое (!)
      float g = 0;
      for (size_t d = 0; d <= negative; ++d)
      {
        if (d == 0) // на первой итерации рассматриваем положительный пример (слово, предсказываемое по контексту)
        {
          target = le.word;
          label = 1;
        }
        else // на остальных итерациях рассматриваем отрицательные примеры (случайные слова из noise distribution)
        {
          next_random_ns = next_random_ns * (unsigned long long)25214903917 + 11;
          target = table[(next_random_ns >> 16) % table_size];
          if (target == 0) target = next_random_ns % (in_vocabulary->size() - 1) + 1;
          if (target == le.word) continue;
          label = 0;
        }
        // вычисляем смещение вектора, соответствующего очередному положительному/отрицательному примеру
        float *targetVectorPtr = syn1 + target * layer1_size;
        // в skip-gram выход скрытого слоя в точности соответствует вектору контекста
        // вычисляем выход нейрона выходного слоя (нейрона, соответствующего рассматриваемому положительному/отрицательному примеру) (hidden -> output)
        float f = std::inner_product(ctxVectorPtr, ctxVectorPtr+layer1_size, targetVectorPtr, 0.0);
        // вычислим градиент умноженный на коэффициент скорости обучения
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        // Propagate errors output -> hidden
        std::transform(neu1e, neu1e+layer1_size, targetVectorPtr, neu1e, [g](float a, float b) -> float {return a + g*b;});
        // Learn weights hidden -> output
        std::transform(targetVectorPtr, targetVectorPtr+layer1_size, ctxVectorPtr, targetVectorPtr, [g](float a, float b) -> float {return a + g*b;});
      } // for all samples
      // Learn weights input -> hidden
      std::transform(ctxVectorPtr, ctxVectorPtr+layer1_size, neu1e, ctxVectorPtr, std::plus<float>());
    } // for all contexts
  } // method-end
};


#endif /* SRC_SG_TRAINER_MIKOLOV_H_ */
