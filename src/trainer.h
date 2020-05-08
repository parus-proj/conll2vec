#ifndef TRAINER_H_
#define TRAINER_H_

#include "learning_example_provider.h"
#include "vocabulary.h"

#include <memory>
#include <string>
#include <chrono>
#include <iostream>
#include <cmath>

#ifdef _MSC_VER
  #define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)
  #define free_aligned(p) _aligned_free((p))
#else
  #define free_aligned(p) free((p))
#endif


#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6


// хранит общие параметры и данные для всех потоков
// реализует логику обучения
class Trainer
{
public:
  // конструктор
  Trainer( std::shared_ptr< LearningExampleProvider> learning_example_provider,
                 std::shared_ptr< CustomVocabulary > words_vocabulary,
                 std::shared_ptr< CustomVocabulary > dep_contexts_vocabulary,
                 std::shared_ptr< CustomVocabulary > assoc_contexts_vocabulary,
                 size_t embedding_dep_size,
                 size_t embedding_assoc_size,
                 size_t epochs,
                 float learning_rate,
                 size_t negative_count )
  : lep(learning_example_provider)
  , w_vocabulary(words_vocabulary)
  , dep_ctx_vocabulary(dep_contexts_vocabulary)
  , assoc_ctx_vocabulary(assoc_contexts_vocabulary)
  , layer1_size(embedding_dep_size + embedding_assoc_size)
  , size_dep(embedding_dep_size)
  , size_assoc(embedding_assoc_size)
  , epoch_count(epochs)
  , alpha(learning_rate)
  , starting_alpha(learning_rate)
  , negative(negative_count)
  , next_random_ns(0)
  {
    // предварительный табличный расчет для логистической функции
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    for (size_t i = 0; i < EXP_TABLE_SIZE; i++) {
      expTable[i] = std::exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
      expTable[i] = expTable[i] / (expTable[i] + 1);                         // Precompute f(x) = x / (x + 1)
    }
    // запомним количество обучающих примеров
    train_words = w_vocabulary->cn_sum();
    // инициализируем распределения, имитирующие шум (для словарей контекстов)
    InitUnigramTable(table_dep, dep_ctx_vocabulary);
    InitUnigramTable(table_assoc, assoc_ctx_vocabulary);
  }
  // деструктор
  virtual ~Trainer()
  {
    free(expTable);
    if (syn0)
      free_aligned(syn0);
    if (syn1_dep)
      free_aligned(syn1_dep);
    if (syn1_assoc)
      free_aligned(syn1_assoc);
    if (table_dep)
      free(table_dep);
    if (table_assoc)
      free(table_assoc);
  }
  // функция инициализации нейросети
  void init_net()
  {
    size_t w_vocab_size = w_vocabulary->size();
    size_t dep_vocab_size = dep_ctx_vocabulary->size();
    size_t assoc_vocab_size = assoc_ctx_vocabulary->size();
    long long ap;
    unsigned long long next_random = 1;

    ap = posix_memalign((void **)&syn0, 128, (long long)w_vocab_size * layer1_size * sizeof(float));
    if (syn0 == nullptr || ap != 0) {std::cerr << "Memory allocation failed" << std::endl; exit(1);}
    for (size_t a = 0; a < w_vocab_size; ++a)
      for (size_t b = 0; b < layer1_size; ++b)
      {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
      }

    ap = posix_memalign((void **)&syn1_dep, 128, (long long)dep_vocab_size * size_dep * sizeof(float));
    if (syn1_dep == nullptr || ap != 0) {std::cerr << "Memory allocation failed" << std::endl; exit(1);}
    std::fill(syn1_dep, syn1_dep+dep_vocab_size*size_dep, 0.0);

    ap = posix_memalign((void **)&syn1_assoc, 128, (long long)assoc_vocab_size * layer1_size * sizeof(float));
    if (syn1_assoc == nullptr || ap != 0) {std::cerr << "Memory allocation failed" << std::endl; exit(1);}
    std::fill(syn1_assoc, syn1_assoc+assoc_vocab_size*size_assoc, 0.0);

    start_learning_tp = std::chrono::steady_clock::now();
  } // method-end
  // обобщенная процедура обучения (точка входа для потоков)
  void train_entry_point( size_t thread_idx )
  {
    next_random_ns = thread_idx;
    // выделение памяти для хранения величины ошибки
    float *neu1e = (float *)calloc(layer1_size, sizeof(float));
    // цикл по эпохам
    for (size_t epochIdx = 0; epochIdx < epoch_count; ++epochIdx)
    {
      if ( !lep->epoch_prepare(thread_idx) )
        return;
      long long word_count = 0, last_word_count = 0;
      // цикл по словам
      while (true)
      {
        // вывод прогресс-сообщений
        // и корректировка коэффициента скорости обучения (alpha)
        if (word_count - last_word_count > 10000)
        {
          word_count_actual += (word_count - last_word_count);
          last_word_count = word_count;
          //if ( debug_mode > 1 )
          {
            std::chrono::steady_clock::time_point current_learning_tp = std::chrono::steady_clock::now();
            std::chrono::duration< double, std::ratio<1> > learning_seconds = current_learning_tp - start_learning_tp;
            printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk  ", 13, alpha,
              word_count_actual / (float)(epoch_count * train_words + 1) * 100,
              word_count_actual / (learning_seconds.count() * 1000) );
            fflush(stdout);
          }
          alpha = starting_alpha * (1 - word_count_actual / (float)(epoch_count * train_words + 1));
          if ( alpha < starting_alpha * 0.0001 )
            alpha = starting_alpha * 0.0001;
        }
        // читаем очередной обучающий пример
        auto learning_example = lep->get(thread_idx);
        word_count = lep->getWordsCount(thread_idx);
        if (!learning_example) break; // признак окончания эпохи (все обучающие примеры перебраны)
        // используем обучающий пример для обучения нейросети
        //todo: сделать различение между обучением основного словаря и словаря собственных имен
        skip_gram( learning_example.value(), neu1e );
      } // for all learning examples
      word_count_actual += (word_count - last_word_count);
      if ( !lep->epoch_unprepare(thread_idx) )
        return;
    } // for all epochs
    free(neu1e);
  } // method-end: train_entry_point
  // функция, реализующая сохранение эмбеддингов
  void saveEmbeddings(const std::string& filename) const
  {
    FILE *fo = fopen(filename.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", w_vocabulary->size(), layer1_size);
    saveEmbeddingsBin_helper(fo, w_vocabulary, syn0);
    fclose(fo);
  } // method-end
  // функция сохранения обоих весовых матриц в файл
  void backup(const std::string& filename) const
  {
    FILE *fo = fopen(filename.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", w_vocabulary->size(), layer1_size);
    // сохраняем весовую матрицу между входным и скрытым слоем
    saveEmbeddingsBin_helper(fo, w_vocabulary, syn0);
    // сохраняем весовую матрицу между скрытым и выходным слоем
    fprintf(fo, "%lu %lu\n", dep_ctx_vocabulary->size(), size_dep);
    saveEmbeddingsBin_helper(fo, dep_ctx_vocabulary, syn1_dep);
    fprintf(fo, "%lu %lu\n", assoc_ctx_vocabulary->size(), size_assoc);
    saveEmbeddingsBin_helper(fo, assoc_ctx_vocabulary, syn1_assoc);
    fclose(fo);
  } // method-end

private:
  std::shared_ptr< LearningExampleProvider > lep;
  std::shared_ptr< CustomVocabulary > w_vocabulary;
  std::shared_ptr< CustomVocabulary > dep_ctx_vocabulary;
  std::shared_ptr< CustomVocabulary > assoc_ctx_vocabulary;
  // размерность скрытого слоя (она же размерность эмбеддинга)
  size_t layer1_size;
  // размерность части эмбеддинга, обучаемого на синтаксических контекстах
  size_t size_dep;
  // размерность части эмбеддинга, обучаемого на ассоциативных контекстах
  size_t size_assoc;
  // количество эпох обучения
  size_t epoch_count;
  // learning rate
  float alpha;
  // начальный learning rate
  float starting_alpha;
  // количество отрицательных примеров на каждый положительный при оптимизации методом negative sampling
  size_t negative;
  // матрицы весов между слоями input-hidden и hidden-output
  float *syn0 = nullptr, *syn1_dep = nullptr, *syn1_assoc = nullptr;
  // табличное представление логистической функции в области определения [-MAX_EXP; +MAX_EXP]
  float *expTable = nullptr;
  // noise distribution for negative sampling
  const size_t table_size = 1e8; // 100 млн.
  int *table_dep = nullptr,
      *table_assoc = nullptr;
  unsigned long long next_random_ns;

  inline void update_random_ns()
  {
    next_random_ns = next_random_ns * (unsigned long long)25214903917 + 11;
  }
  // функция инициализации распределения, имитирующего шум, для метода оптимизации negative sampling
  void InitUnigramTable(int*& table, std::shared_ptr< CustomVocabulary > vocabulary)
  {
    double train_words_pow = 0;
    double d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    // вычисляем нормирующую сумму (за слагаемое берется абсолютная частота слова/контекста в степени 3/4)
    for (size_t a = 0; a < vocabulary->size(); ++a)
      train_words_pow += std::pow(vocabulary->idx_to_data(a).cn, power);
    // заполняем таблицу распределения, имитирующего шум
    size_t i = 0;
    d1 = std::pow(vocabulary->idx_to_data(i).cn, power) / train_words_pow;
    for (size_t a = 0; a < table_size; ++a)
    {
      table[a] = i;
      if (a / (double)table_size > d1)
      {
        i++;
        d1 += std::pow(vocabulary->idx_to_data(i).cn, power) / train_words_pow;
      }
      if (i >= vocabulary->size())
        i = vocabulary->size() - 1;
    }
  } // method-end
  // функция, реализующая модель обучения skip-gram
  void skip_gram(const LearningExample& le, float *neu1e )
  {
    // вычисляем смещение вектора, соответствующего целевому слову
    float *targetVectorPtr = syn0 + le.word * layer1_size;
    // цикл по синтаксическим контекстам
    for (auto&& ctx_idx : le.dep_context)
    {
      // зануляем текущие значения ошибок (это частная производная ошибки E по выходу скрытого слоя h)
      std::fill(neu1e, neu1e+size_dep, 0.0);
      size_t selected_ctx;
      int label;     // знаковое целое (!)
      float g = 0;
      for (size_t d = 0; d <= negative; ++d)
      {
        if (d == 0) // на первой итерации рассматриваем положительный пример (контекст)
        {
          selected_ctx = ctx_idx;
          label = 1;
        }
        else // на остальных итерациях рассматриваем отрицательные примеры (случайные контексты из noise distribution)
        {
          update_random_ns();
          selected_ctx = table_dep[(next_random_ns >> 16) % table_size];
          label = 0;
        }
        // вычисляем смещение вектора, соответствующего очередному положительному/отрицательному примеру
        float *ctxVectorPtr = syn1_dep + selected_ctx * size_dep;
        // в skip-gram выход скрытого слоя в точности соответствует вектору целевого слова
        // вычисляем выход нейрона выходного слоя (нейрона, соответствующего рассматриваемому положительному/отрицательному примеру) (hidden -> output)
        float f = std::inner_product(targetVectorPtr, targetVectorPtr+size_dep, ctxVectorPtr, 0.0);
        // вычислим градиент умноженный на коэффициент скорости обучения
        if      (f > MAX_EXP)  g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else                   g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        // Propagate errors output -> hidden
        std::transform(neu1e, neu1e+size_dep, ctxVectorPtr, neu1e, [g](float a, float b) -> float {return a + g*b;});
        // Learn weights hidden -> output
        std::transform(ctxVectorPtr, ctxVectorPtr+size_dep, targetVectorPtr, ctxVectorPtr, [g](float a, float b) -> float {return a + g*b;});
      } // for all samples
      // Learn weights input -> hidden
      std::transform(targetVectorPtr, targetVectorPtr+size_dep, neu1e, targetVectorPtr, std::plus<float>());
    } // for all dep contexts
    // цикл по ассоциативным контекстам
    targetVectorPtr += size_dep; // используем оставшуюся часть вектора для ассоциаций
    neu1e += size_dep;
    for (auto&& ctx_idx : le.assoc_context)
    {
      // зануляем текущие значения ошибок (это частная производная ошибки E по выходу скрытого слоя h)
      std::fill(neu1e, neu1e+size_assoc, 0.0);
      size_t selected_ctx;
      int label;     // знаковое целое (!)
      float g = 0;
      for (size_t d = 0; d <= negative; ++d)
      {
        if (d == 0) // на первой итерации рассматриваем положительный пример (контекст)
        {
          selected_ctx = ctx_idx;
          label = 1;
        }
        else // на остальных итерациях рассматриваем отрицательные примеры (случайные контексты из noise distribution)
        {
          update_random_ns();
          selected_ctx = table_assoc[(next_random_ns >> 16) % table_size];
          label = 0;
        }
        // вычисляем смещение вектора, соответствующего очередному положительному/отрицательному примеру
        float *ctxVectorPtr = syn1_assoc + selected_ctx * size_assoc;
        // в skip-gram выход скрытого слоя в точности соответствует вектору целевого слова
        // вычисляем выход нейрона выходного слоя (нейрона, соответствующего рассматриваемому положительному/отрицательному примеру) (hidden -> output)
        float f = std::inner_product(targetVectorPtr, targetVectorPtr+size_assoc, ctxVectorPtr, 0.0);
        // вычислим градиент умноженный на коэффициент скорости обучения
        if      (f > MAX_EXP)  g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else                   g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        // Propagate errors output -> hidden
        std::transform(neu1e, neu1e+size_assoc, ctxVectorPtr, neu1e, [g](float a, float b) -> float {return a + g*b;});
        // Learn weights hidden -> output
        std::transform(ctxVectorPtr, ctxVectorPtr+size_assoc, targetVectorPtr, ctxVectorPtr, [g](float a, float b) -> float {return a + g*b;});
      } // for all samples
      // Learn weights input -> hidden
      std::transform(targetVectorPtr, targetVectorPtr+size_assoc, neu1e, targetVectorPtr, std::plus<float>());
    } // for all assoc contexts
  } // method-end
private:
  uint64_t train_words = 0;
  uint64_t word_count_actual = 0;
  std::chrono::steady_clock::time_point start_learning_tp;

  void saveEmbeddingsBin_helper(FILE *fo, std::shared_ptr< CustomVocabulary > vocabulary, float *weight_matrix) const
  {
    for (size_t a = 0; a < vocabulary->size(); ++a)
    {
      fprintf(fo, "%s ", vocabulary->idx_to_data(a).word.c_str());
      for (size_t b = 0; b < layer1_size; ++b)
        fwrite(&weight_matrix[a * layer1_size + b], sizeof(float), 1, fo);
      fprintf(fo, "\n");
    }
  }
  void saveEmbeddingsTxt_helper(FILE *fo, std::shared_ptr< CustomVocabulary > vocabulary, float *weight_matrix) const
  {
    for (size_t a = 0; a < vocabulary->size(); ++a)
    {
      fprintf(fo, "%s", vocabulary->idx_to_data(a).word.c_str());
      for (size_t b = 0; b < layer1_size; ++b)
        fprintf(fo, " %lf", weight_matrix[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  }
}; // class-decl-end


#endif /* TRAINER_H_ */
