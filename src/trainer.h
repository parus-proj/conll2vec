#ifndef TRAINER_H_
#define TRAINER_H_

#include "learning_example_provider.h"
#include "vocabulary.h"
#include "original_word2vec_vocabulary.h"
#include "vectors_model.h"

#include <memory>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>

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
           bool trainProperNames,
           std::shared_ptr< CustomVocabulary > dep_contexts_vocabulary,
           std::shared_ptr< CustomVocabulary > assoc_contexts_vocabulary,
           size_t embedding_dep_size,
           size_t embedding_assoc_size,
           size_t epochs,
           float learning_rate,
           size_t negative_count,
           size_t total_threads_count )
  : lep(learning_example_provider)
  , w_vocabulary(words_vocabulary)
  , w_vocabulary_size(words_vocabulary->size())
  , proper_names(trainProperNames)
  , dep_ctx_vocabulary(dep_contexts_vocabulary)
  , assoc_ctx_vocabulary(assoc_contexts_vocabulary)
  , layer1_size(embedding_dep_size + embedding_assoc_size)
  , size_dep(embedding_dep_size)
  , size_assoc(embedding_assoc_size)
  , epoch_count(epochs)
  , alpha(learning_rate)
  , starting_alpha(learning_rate)
  , negative(negative_count)
  {
    // предварительный табличный расчет для логистической функции
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    for (size_t i = 0; i < EXP_TABLE_SIZE; i++) {
      expTable[i] = std::exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
      expTable[i] = expTable[i] / (expTable[i] + 1);                         // Precompute f(x) = x / (x + 1)
    }
    // запомним количество обучающих примеров
    train_words = w_vocabulary->cn_sum();
    // настроим периодичность обновления "коэффициента скорости обучения"
    alpha_chunk = (train_words - 1) / total_threads_count;
    if (alpha_chunk > 10000)
      alpha_chunk = 10000;
    // инициализируем распределения, имитирующие шум (для словарей контекстов)
    if ( dep_ctx_vocabulary )
      InitUnigramTable(table_dep, dep_ctx_vocabulary);
    if ( assoc_ctx_vocabulary )
//      InitUnigramTable(table_assoc, assoc_ctx_vocabulary);
// FOR-RESEARCH-begin
      InitUnigramTable(table_assoc, w_vocabulary);
// FOR-RESEARCH-end
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
  // функция создания весовых матриц нейросети
  void create_net()
  {
    long long ap = 0;

    size_t w_vocab_size = w_vocabulary->size();
    ap = posix_memalign((void **)&syn0, 128, (long long)w_vocab_size * layer1_size * sizeof(float));
    if (syn0 == nullptr || ap != 0) {std::cerr << "Memory allocation failed" << std::endl; exit(1);}

    if ( dep_ctx_vocabulary )
    {
      size_t dep_vocab_size = dep_ctx_vocabulary->size();
      ap = posix_memalign((void **)&syn1_dep, 128, (long long)dep_vocab_size * size_dep * sizeof(float));
      if (syn1_dep == nullptr || ap != 0) {std::cerr << "Memory allocation failed" << std::endl; exit(1);}
    }
    if ( assoc_ctx_vocabulary )
    {
      size_t assoc_vocab_size = assoc_ctx_vocabulary->size();
      ap = posix_memalign((void **)&syn1_assoc, 128, (long long)assoc_vocab_size * size_assoc * sizeof(float));
      if (syn1_assoc == nullptr || ap != 0) {std::cerr << "Memory allocation failed" << std::endl; exit(1);}
    }
  } // method-end
  // функция инициализации нейросети
  void init_net()
  {
    unsigned long long next_random = 1;
    size_t w_vocab_size = w_vocabulary->size();
//    for (size_t a = 0; a < w_vocab_size; ++a)
//      for (size_t b = 0; b < layer1_size; ++b)
//      {
//        next_random = next_random * (unsigned long long)25214903917 + 11;
//        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
//      }
    for (size_t a = 0; a < w_vocab_size; ++a)
    {
      float denominator = std::sqrt(w_vocabulary->idx_to_data(a).cn);
      for (size_t b = 0; b < layer1_size; ++b)
      {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / denominator; // более частотные ближе к нулю
      }
    }

    if ( dep_ctx_vocabulary )
    {
      size_t dep_vocab_size = dep_ctx_vocabulary->size();
      std::fill(syn1_dep, syn1_dep+dep_vocab_size*size_dep, 0.0);
    }

    if ( assoc_ctx_vocabulary )
    {
      size_t assoc_vocab_size = assoc_ctx_vocabulary->size();
      std::fill(syn1_assoc, syn1_assoc+assoc_vocab_size*size_assoc, 0.0);
    }

    start_learning_tp = std::chrono::steady_clock::now();
  } // method-end
  // обобщенная процедура обучения (точка входа для потоков)
  void train_entry_point( size_t thread_idx )
  {
//    if ( thread_idx == 0 )
//      ctrl_log_prepare();
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
        if (word_count - last_word_count > alpha_chunk)
        {
          word_count_actual += (word_count - last_word_count);
          last_word_count = word_count;
          fraction = word_count_actual / (float)(epoch_count * train_words + 1);
          //if ( debug_mode != 0 )
          {
            std::chrono::steady_clock::time_point current_learning_tp = std::chrono::steady_clock::now();
            std::chrono::duration< double, std::ratio<1> > learning_seconds = current_learning_tp - start_learning_tp;
            printf( "\rAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk   ", alpha,
                    fraction * 100,
                    word_count_actual / (learning_seconds.count() * 1000) );
//            printf( "Ctrl: %.5f        ", ctrl_log_calc_not_sim() );
            fflush(stdout);
          }
          alpha = starting_alpha * (1.0 - fraction);
          if ( alpha < starting_alpha * 0.0001 )
            alpha = starting_alpha * 0.0001;
        } // if ('checkpoint')
        // читаем очередной обучающий пример
        auto learning_example = lep->get(thread_idx);
        word_count = lep->getWordsCount(thread_idx);
        if (!learning_example) break; // признак окончания эпохи (все обучающие примеры перебраны)
        // используем обучающий пример для обучения нейросети
        skip_gram( learning_example.value(), neu1e );
      } // for all learning examples
      word_count_actual += (word_count - last_word_count);
      if ( !lep->epoch_unprepare(thread_idx) )
        return;
    } // for all epochs
    free(neu1e);
  } // method-end: train_entry_point
  // функция, реализующая сохранение эмбеддингов
  void saveEmbeddings(const std::string& filename, bool useTxtFmt = false) const
  {
    FILE *fo = fopen(filename.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", w_vocabulary->size(), layer1_size);
    if ( !useTxtFmt )
      saveEmbeddingsBin_helper(fo, w_vocabulary, syn0, layer1_size);
    else
      saveEmbeddingsTxt_helper(fo, w_vocabulary, syn0, layer1_size);
    fclose(fo);
  } // method-end
  // функция добавления эмбеддингов в уже существующую модель
  void appendEmbeddings(const std::string& filename, bool useTxtFmt = false) const
  {
    // загружаем всю модель в память
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    if ( !ifs.good() ) { std::cerr << "Append: Model file not found" << std::endl; return; }
    std::string buf;
    size_t old_vocab_size, emb_size;
    ifs >> old_vocab_size;
    ifs >> emb_size;
    std::getline(ifs,buf); // считываем конец строки
    if (emb_size != layer1_size) { std::cerr << "Append: Dimensions fail" << std::endl; return; }
    std::shared_ptr< CustomVocabulary > old_vocab = std::make_shared<OriginalWord2VecVocabulary>();
    float *old_matrix;
    long long ap = posix_memalign((void **)&old_matrix, 128, (long long)old_vocab_size * emb_size * sizeof(float));
    if (old_matrix == nullptr || ap != 0) {std::cerr << "Append: Memory allocation failed" << std::endl; exit(1);}
    for (size_t i = 0; i < old_vocab_size; ++i)
    {
      std::getline(ifs, buf, ' '); // читаем слово (до пробела)
      old_vocab->append( buf, 0 );
      float* eOffset = old_matrix + i*emb_size;
      if ( !useTxtFmt )
        ifs.read( reinterpret_cast<char*>( eOffset ), sizeof(float)*emb_size );
      else
      {
        for (size_t j = 0; j < emb_size; ++j)
          ifs >> eOffset[j];
      }
      std::getline(ifs,buf); // считываем конец строки
    }
    ifs.close();
    // сохраняем старую модель, затем текущую
    FILE *fo = fopen(filename.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", old_vocab_size + w_vocabulary->size(), emb_size);
    if ( !useTxtFmt )
      saveEmbeddingsBin_helper(fo, old_vocab, old_matrix, emb_size);
    else
      saveEmbeddingsTxt_helper(fo, old_vocab, old_matrix, emb_size);
    free_aligned(old_matrix);
    if ( !useTxtFmt )
      saveEmbeddingsBin_helper(fo, w_vocabulary, syn0, emb_size);
    else
      saveEmbeddingsTxt_helper(fo, w_vocabulary, syn0, emb_size);
    fclose(fo);
  } // method-end
  // функция сохранения весовых матриц в файл
  void backup(const std::string& filename, bool left = true, bool right= true) const
  {
    FILE *fo = fopen(filename.c_str(), "wb");
    // сохраняем весовую матрицу между входным и скрытым слоем
    if (left)
    {
      fprintf(fo, "%lu %lu\n", w_vocabulary->size(), layer1_size);
      saveEmbeddingsBin_helper(fo, w_vocabulary, syn0, layer1_size);
    }
    // сохраняем весовые матрицы между скрытым и выходным слоем
    if (right)
    {
      if ( dep_ctx_vocabulary )
      {
        fprintf(fo, "%lu %lu\n", dep_ctx_vocabulary->size(), size_dep);
        saveEmbeddingsBin_helper(fo, dep_ctx_vocabulary, syn1_dep, size_dep);
      }
      if ( assoc_ctx_vocabulary )
      {
        fprintf(fo, "%lu %lu\n", assoc_ctx_vocabulary->size(), size_assoc);
        saveEmbeddingsBin_helper(fo, assoc_ctx_vocabulary, syn1_assoc, size_assoc);
      }
    }
    fclose(fo);
  } // method-end
  // функция восстановления весовых матриц из файла (предполагает, что память уже выделена)
  bool restore(const std::string& filename, bool left = true, bool right= true)
  {
    // открываем файл модели
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    if ( !ifs.good() )
    {
      std::cerr << "Restore: Backup file not found" << std::endl;
      return false;
    }
    // загружаем матрицу между входным и скрытым слоем
    if (left)
    {
      size_t vocab_size, emb_size;
      restore__read_sizes(ifs, vocab_size, emb_size);
      if (vocab_size != w_vocabulary->size() || emb_size != layer1_size)
      {
        std::cerr << "Restore: Dimensions fail" << std::endl;
        return false;
      }
      if ( !restore__read_matrix(ifs, w_vocabulary, layer1_size, syn0) )
        return false;
    }
    // загружаем матрицы между скрытым и выходным слоем
    if (right)
    {
      size_t vocab_size, emb_size;
      restore__read_sizes(ifs, vocab_size, emb_size);
      if (vocab_size != dep_ctx_vocabulary->size() || emb_size != size_dep)
      {
        std::cerr << "Restore: Dimensions fail" << std::endl;
        return false;
      }
      if ( !restore__read_matrix(ifs, dep_ctx_vocabulary, size_dep, syn1_dep) )
        return false;

      restore__read_sizes(ifs, vocab_size, emb_size);
      if (vocab_size != assoc_ctx_vocabulary->size() || emb_size != size_assoc)
      {
        std::cerr << "Restore: Dimensions fail" << std::endl;
        return false;
      }
      if ( !restore__read_matrix(ifs, assoc_ctx_vocabulary, size_assoc, syn1_assoc) )
        return false;
    }
    start_learning_tp = std::chrono::steady_clock::now();
    return true;
  } // method-end
  // функция восстановления левой весовой матрицы из векторной модели
  bool restore_left_matrix(const VectorsModel& vm)
  {
    if (layer1_size != vm.emb_size)
    {
      std::cerr << "restore_left_matrix: dimensions discrepancy" << std::endl;
      return false;
    }
    for (size_t w = 0; w < w_vocabulary->size(); ++w)
    {
      auto& voc_rec = w_vocabulary->idx_to_data(w);
      size_t vm_idx = vm.get_word_idx( voc_rec.word );
      if (vm_idx == vm.words_count) // вектора неизвестных слов остаются случайно-инициализированными
      {
        //std::cerr << "warning: vector representation random init: " << voc_rec.word << std::endl;
        continue;
      }
      float* hereOffset  = syn0 + w * layer1_size;
      float* thereOffset = vm.embeddings + vm_idx * vm.emb_size;
      std::copy(thereOffset, thereOffset + vm.emb_size, hereOffset);
    }
    return true;
  } // method-end

private:
  std::shared_ptr< LearningExampleProvider > lep;
  std::shared_ptr< CustomVocabulary > w_vocabulary;
  size_t w_vocabulary_size;
  bool proper_names;  // признак того, что выполняется обучение векторных представлений для собственных имен
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
  unsigned long long next_random_ns = 0;

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
//  void InitUniformTable(int*& table, std::shared_ptr< CustomVocabulary > vocabulary)
//  {
//    table = (int *)malloc(table_size * sizeof(int));
//    size_t i = 0;
//    double d1 = 1.0 / vocabulary->size();
//    for (size_t a = 0; a < table_size; ++a)
//    {
//      table[a] = i;
//      if (a / (double)table_size > d1)
//      {
//        i++;
//        d1 += 1.0 / vocabulary->size();
//      }
//    }
//  } // method-end
  // функция, реализующая модель обучения skip-gram
  void skip_gram( const LearningExample& le, float *neu1e )
  {
    float norm_factor = negative - fraction*(negative-1);
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
        if ( std::isnan(f) ) continue;
        // вычислим градиент умноженный на коэффициент скорости обучения
        g = alpha_error(f, label);
        // Propagate errors output -> hidden
        if (d==0)
          std::transform(neu1e, neu1e+size_dep, ctxVectorPtr, neu1e, [g](float a, float b) -> float {return a + g*b;});
        else
        {
          float g_norm = g/norm_factor;
          std::transform(neu1e, neu1e+size_dep, ctxVectorPtr, neu1e, [g_norm](float a, float b) -> float {return a + g_norm*b;});
        }
        // Learn weights hidden -> output
        if ( !proper_names )
          std::transform(ctxVectorPtr, ctxVectorPtr+size_dep, targetVectorPtr, ctxVectorPtr, [g](float a, float b) -> float {return a + g*b;});
      } // for all samples
      // Learn weights input -> hidden
      //std::transform(targetVectorPtr, targetVectorPtr+size_dep, neu1e, targetVectorPtr, std::plus<float>());
      std::transform(targetVectorPtr, targetVectorPtr+size_dep, neu1e, targetVectorPtr,
                     [](float a, float b) -> float
                     {
                       if ( a*b < 0 ) return a + b; // разнознаковые
                       const float TH = 0.5;
                       float abs_a = fabs(a);
                       if (abs_a <= TH)
                         return a + b;
                       else
                         return a + b / (abs_a+TH);
                     }
                    );
    } // for all dep contexts

//    // цикл по ассоциативным контекстам
//    targetVectorPtr += size_dep; // используем оставшуюся часть вектора для ассоциаций
//    neu1e += size_dep;
//    for (auto&& ctx_idx : le.assoc_context)
//    {
//      // зануляем текущие значения ошибок (это частная производная ошибки E по выходу скрытого слоя h)
//      std::fill(neu1e, neu1e+size_assoc, 0.0);
//      size_t selected_ctx;
//      int label;     // знаковое целое (!)
//      float g = 0;
//      for (size_t d = 0; d <= negative; ++d)
//      {
//        if (d == 0) // на первой итерации рассматриваем положительный пример (контекст)
//        {
//          selected_ctx = ctx_idx;
//          label = 1;
//        }
//        else // на остальных итерациях рассматриваем отрицательные примеры (случайные контексты из noise distribution)
//        {
//          update_random_ns();
//          selected_ctx = table_assoc[(next_random_ns >> 16) % table_size];
//          label = 0;
//        }
//        // вычисляем смещение вектора, соответствующего очередному положительному/отрицательному примеру
//        float *ctxVectorPtr = syn1_assoc + selected_ctx * size_assoc;
//        // в skip-gram выход скрытого слоя в точности соответствует вектору целевого слова
//        // вычисляем выход нейрона выходного слоя (нейрона, соответствующего рассматриваемому положительному/отрицательному примеру) (hidden -> output)
//        float f = std::inner_product(targetVectorPtr, targetVectorPtr+size_assoc, ctxVectorPtr, 0.0);
//        if ( std::isnan(f) ) continue;
//        // вычислим градиент умноженный на коэффициент скорости обучения
//        if      (f > MAX_EXP)  g = (label - 1) * alpha;
//        else if (f < -MAX_EXP) g = (label - 0) * alpha;
//        else                   g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
//        // Propagate errors output -> hidden
//        if (d==0)
//         std::transform(neu1e, neu1e+size_assoc, ctxVectorPtr, neu1e, [g](float a, float b) -> float {return a + g*b;});
//        else
//        {
//          float g_norm = g/negative;
//          std::transform(neu1e, neu1e+size_assoc, ctxVectorPtr, neu1e, [g_norm](float a, float b) -> float {return a + g_norm*b;});
//        }
//        // Learn weights hidden -> output
//        if ( !proper_names )
//          std::transform(ctxVectorPtr, ctxVectorPtr+size_assoc, targetVectorPtr, ctxVectorPtr, [g](float a, float b) -> float {return a + g*b;});
//      } // for all samples
//      // Learn weights input -> hidden
//      //std::transform(targetVectorPtr, targetVectorPtr+size_assoc, neu1e, targetVectorPtr, std::plus<float>());
//      std::transform(targetVectorPtr, targetVectorPtr+size_assoc, neu1e, targetVectorPtr,
//                     [](float a, float b) -> float
//                     {
//                       if ( a*b < 0 ) return a + b; // разнознаковые
//                       const float TH = 0.5;
//                       float abs_a = fabs(a);
//                       if (abs_a <= TH)
//                         return a + b;
//                       else
//                         return a + b / (abs_a+TH);
//                     }
//                    );
//    } // for all assoc contexts

// FOR-RESEARCH-begin   -- скорость обучения выше, показатели чуть хуже... (оценка предварительная)
    // цикл по ассоциативным контекстам
    targetVectorPtr += size_dep; // используем оставшуюся часть вектора для ассоциаций
    for (auto&& ctx_idx : le.assoc_context)
    {
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
          //selected_ctx = table_assoc[(next_random_ns >> 16) % table_size]; // unigram distribution
          selected_ctx = (next_random_ns >> 16) % w_vocabulary_size; // uniform distribution
          label = 0;
        }
        // вычисляем смещение вектора, соответствующего очередному положительному/отрицательному примеру
        float *ctxVectorPtr = syn0 + selected_ctx * layer1_size + size_dep;
        // в skip-gram выход скрытого слоя в точности соответствует вектору целевого слова
        // вычисляем выход нейрона выходного слоя (нейрона, соответствующего рассматриваемому положительному/отрицательному примеру) (hidden -> output)
        float f = std::inner_product(targetVectorPtr, targetVectorPtr+size_assoc, ctxVectorPtr, 0.0);
        if ( std::isnan(f) ) continue;
        // вычислим градиент умноженный на коэффициент скорости обучения
        g = alpha_error(f, label);
        // Learn weights
        std::transform(ctxVectorPtr, ctxVectorPtr+size_assoc, targetVectorPtr, ctxVectorPtr, [g](float a, float b) -> float {return a + g*b;});
      } // for all samples
    } // for all assoc contexts
// FOR-RESEARCH-end

  } // method-end

  // вычисление величины ошибки на выходном слое, умноженной на коэффициент скорости обучения
  inline float alpha_error(float f, int label) const
  {
    if      (f > MAX_EXP)  return (label - 1) * alpha;
    else if (f < -MAX_EXP) return (label - 0) * alpha;
    else                   return (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
  } // method-end

private:
  uint64_t train_words = 0;
  uint64_t word_count_actual = 0;
  float fraction = 0.0;
  // периодичность, с которой корректируется "коэф.скорости обучения"
  long long alpha_chunk = 0;
  std::chrono::steady_clock::time_point start_learning_tp;

  void saveEmbeddingsBin_helper(FILE *fo, std::shared_ptr< CustomVocabulary > vocabulary, float *weight_matrix, size_t emb_size) const
  {
    for (size_t a = 0; a < vocabulary->size(); ++a)
    {
      fprintf(fo, "%s ", vocabulary->idx_to_data(a).word.c_str());
      for (size_t b = 0; b < emb_size; ++b)
        fwrite(&weight_matrix[a * emb_size + b], sizeof(float), 1, fo);
      fprintf(fo, "\n");
    }
  } // method-end
  void saveEmbeddingsTxt_helper(FILE *fo, std::shared_ptr< CustomVocabulary > vocabulary, float *weight_matrix, size_t emb_size) const
  {
    for (size_t a = 0; a < vocabulary->size(); ++a)
    {
      fprintf(fo, "%s", vocabulary->idx_to_data(a).word.c_str());
      for (size_t b = 0; b < emb_size; ++b)
        fprintf(fo, " %lf", weight_matrix[a * emb_size + b]);
      fprintf(fo, "\n");
    }
  } // method-end
  void restore__read_sizes(std::ifstream& ifs, size_t& vocab_size, size_t& emb_size)
  {
    std::string buf;
    ifs >> vocab_size;
    ifs >> emb_size;
    std::getline(ifs,buf); // считываем конец строки
  } // method-end
  bool restore__read_matrix(std::ifstream& ifs, std::shared_ptr< CustomVocabulary > vocab, size_t emb_size, float *matrix)
  {
    std::string buf;
    size_t vocab_size = vocab->size();
    for (size_t i = 0; i < vocab_size; ++i)
    {
      std::getline(ifs, buf, ' '); // читаем слово (до пробела)
      if ( vocab->idx_to_data(i).word != buf )
      {
        std::cerr << "Restore: Vocabulary divergence" << std::endl;
        return false;
      }
      float* eOffset = matrix + i*emb_size;
      ifs.read( reinterpret_cast<char*>( eOffset ), sizeof(float)*emb_size );
      std::getline(ifs,buf); // считываем конец строки
    }
    return true;
  } // method-end

//private:
//  // множество несвязанных слов (для отладочной индикации средней меры сходства между ними в ходе обучения)
//  std::vector<std::pair<size_t, size_t>> dbg_not_sim;
//  void ctrl_log_prepare()
//  {
//    const std::vector<std::pair<std::string, std::string>> dbg_not_sim_strs = {
//        {"синий", "кофе"},
//        {"синий", "сообщать"},
//        {"синий", "быстро"},
//        {"синий", "восемь"},
//        {"кофе", "сообщать"},
//        {"кофе", "высоко"},
//        {"кофе", "восемь"},
//        {"сообщать", "высоко"},
//        {"сообщать", "восемь"},
//        {"быстро", "восемь"}
//      };
//    for (auto& p : dbg_not_sim_strs)
//    {
//      size_t i1 = w_vocabulary->word_to_idx(p.first);
//      size_t i2 = w_vocabulary->word_to_idx(p.second);
//      if (i1 == std::numeric_limits<size_t>::max() || i2 == std::numeric_limits<size_t>::max()) // неизвестные слова
//        continue;
//      dbg_not_sim.push_back( std::make_pair(i1, i2) );
//    }
//  } // method-end
//  float ctrl_log_calc_not_sim()
//  {
//    float result = 0;
//    size_t cnt = 0;
//    for (auto& p : dbg_not_sim)
//    {
//      float* o1 = syn0 + p.first * layer1_size;
//      float* o2 = syn0 + p.second * layer1_size;
//      float ip = std::inner_product(o1, o1+layer1_size, o2, 0.0);
//      float len1 = std::sqrt( std::inner_product(o1, o1+layer1_size, o1, 0.0) );
//      float len2 = std::sqrt( std::inner_product(o2, o2+layer1_size, o2, 0.0) );
//      if ( !std::isnormal(len1) || !std::isnormal(len2) || !std::isnormal(ip) )
//        continue;
//      result += ip / len1 / len2;
//      ++cnt;
//    }
//    return result/cnt;
//  } // method-end
}; // class-decl-end


#endif /* TRAINER_H_ */
