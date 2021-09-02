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
           size_t embedding_gramm_size,
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
  , size_gramm(embedding_gramm_size)
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
    if ( assoc_ctx_vocabulary && proper_names )
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

    if ( assoc_ctx_vocabulary && proper_names )
    {
      size_t assoc_vocab_size = assoc_ctx_vocabulary->size();
      std::fill(syn1_assoc, syn1_assoc+assoc_vocab_size*size_assoc, 0.0);
    }

    start_learning_tp = std::chrono::steady_clock::now();
  } // method-end
  void create_and_init_gramm_net()
  {
    long long ap = 0;
    unsigned long long next_random = 1;

    size_t w_vocab_size = w_vocabulary->size();
    ap = posix_memalign((void **)&syn0, 128, (long long)w_vocab_size * size_gramm * sizeof(float));
    if (syn0 == nullptr || ap != 0) {std::cerr << "Memory allocation failed" << std::endl; exit(1);}
    for (size_t a = 0; a < w_vocab_size; ++a)
      for (size_t b = 0; b < size_gramm; ++b)
      {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * size_gramm + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / 100;
      }

    size_t output_size = lep->getGrammemesVectorSize();
    ap = posix_memalign((void **)&syn1_assoc, 128, (long long)output_size * size_gramm * sizeof(float));
    if (syn1_assoc == nullptr || ap != 0) {std::cerr << "Memory allocation failed" << std::endl; exit(1);}
    std::fill(syn1_assoc, syn1_assoc+output_size*size_gramm, 0.0);

    start_learning_tp = std::chrono::steady_clock::now();
  } // method-end
  // обобщенная процедура обучения (точка входа для потоков)
  void train_entry_point( size_t thread_idx )
  {
    unsigned long long next_random_ns = thread_idx;
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
        skip_gram( learning_example.value(), neu1e, next_random_ns );
      } // for all learning examples
      word_count_actual += (word_count - last_word_count);
      if ( !lep->epoch_unprepare(thread_idx) )
        return;
    } // for all epochs
    free(neu1e);
  } // method-end: train_entry_point
  // процедура обучения грамматического вектора (точка входа для потоков)
  void train_entry_point__gramm( size_t thread_idx )
  {
    // выделение памяти для хранения выхода нейросети и дельт нейронов
    size_t output_size = lep->getGrammemesVectorSize();
    float *y = (float *)calloc(output_size, sizeof(float));
    float *eo = (float *)calloc(output_size, sizeof(float));
    float *eh = (float *)calloc(size_gramm, sizeof(float));
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
            fflush(stdout);
          }
          alpha = starting_alpha * (1.0 - fraction);
          if ( alpha < starting_alpha * 0.0001 )
            alpha = starting_alpha * 0.0001;
        } // if ('checkpoint')
        // читаем очередной обучающий пример
        auto learning_example = lep->get(thread_idx, true);
        word_count = lep->getWordsCount(thread_idx);
        if (!learning_example) break; // признак окончания эпохи (все обучающие примеры перебраны)
        // используем обучающий пример для обучения нейросети

        // прямой проход
        float* wordVectorPtr= syn0 + learning_example->word * size_gramm;
        for (size_t g = 0; g < output_size; ++g)
        {
          float* offs = syn1_assoc + g  * size_gramm;
          y[g] = sigmoid( std::inner_product(wordVectorPtr, wordVectorPtr+size_gramm, offs, 0.0) );
        }
        //softmax(y, output_size);

        // обратный проход
        std::fill(eh, eh+size_gramm, 0.0);
        //float tSum = std::accumulate(learning_example->assoc_context.begin(), learning_example->assoc_context.end(), 0);
        std::transform(learning_example->assoc_context.begin(), learning_example->assoc_context.end(), y, eo, [this](float a, float b) -> float {return (a - b) * alpha;});
        // преобразуем вторую матрицу и попутно копим дельты для скрытого слоя
        for (size_t g = 0; g < output_size; ++g)
          for (size_t i = 0; i < size_gramm; ++i)
          {
            eh[i] += syn1_assoc[g*size_gramm+i] * eo[g];
            syn1_assoc[g*size_gramm+i] += wordVectorPtr[i] * eo[g];
          }
        // преобразуем первую матрицу
        std::transform(wordVectorPtr, wordVectorPtr+size_gramm, eh, wordVectorPtr, std::plus<float>());

      } // for all learning examples
      word_count_actual += (word_count - last_word_count);
      if ( !lep->epoch_unprepare(thread_idx) )
        return;
    } // for all epochs
    free(y);
    free(eo);
    free(eh);
  } // method-end: train_entry_point__gramm
  void saveGrammaticalEmbeddings(const VectorsModel& vm, float g_ratio, const std::string& oov_voc_fn, const std::string& filename, bool useTxtFmt = false) const
  {
    const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
    // создаем грамматическое представление -- заглушку (будем приписывать её словам, для которых такие представления не тренировались)
    float *stub = (float *)calloc(size_gramm, sizeof(float));
    std::fill(stub, stub+size_gramm, 1e-15);
    stub[0] = g_ratio;
    // масштабируем абсолютные значения грамматических векторов
    size_t syn0size = w_vocabulary->size() * size_gramm;
    for (size_t i = 0; i < syn0size; ++i)
      syn0[i] *= g_ratio;
    // загружаем словарь oov-суффиксов (если это требуется)
    std::shared_ptr< OriginalWord2VecVocabulary > v_oov = oov_voc_fn.empty() ? nullptr : std::make_shared<OriginalWord2VecVocabulary>();
    if ( v_oov && !v_oov->load( oov_voc_fn ) ) // fatal
      return;
    // дописываем грамм-вектора к семантическим
    FILE *fo = fopen(filename.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", vm.vocab.size() + (v_oov ? v_oov->size() : 0), vm.emb_size + size_gramm);
    for (size_t a = 0; a < vm.vocab.size(); ++a)
    {
      VectorsModel::write_embedding__start(fo, useTxtFmt, vm.vocab[a]);
      VectorsModel::write_embedding__vec(fo, useTxtFmt, &vm.embeddings[a*vm.emb_size], 0, vm.emb_size);
      size_t tok_idx = w_vocabulary->word_to_idx(vm.vocab[a]);
      if ( tok_idx != INVALID_IDX )
        VectorsModel::write_embedding__vec(fo, useTxtFmt, &syn0[tok_idx * size_gramm], 0, size_gramm);
      else
        VectorsModel::write_embedding__vec(fo, useTxtFmt, stub, 0, size_gramm);
      VectorsModel::write_embedding__fin(fo);
    }
    if (v_oov)
    {
      // создаем опорный вектор для OOV (семантическая часть)
      float *support_oov_embedding = (float *) malloc(vm.emb_size*sizeof(float));
      calc_support_embedding(vm.words_count, vm.emb_size, vm.embeddings, support_oov_embedding);
      auto toks_cnt = w_vocabulary->size() - v_oov->size();
      for (size_t a = 0; a < v_oov->size(); ++a)
      {
        VectorsModel::write_embedding__start(fo, useTxtFmt, v_oov->idx_to_data(a).word);
        VectorsModel::write_embedding__vec(fo, useTxtFmt, support_oov_embedding, 0, vm.emb_size);
        VectorsModel::write_embedding__vec(fo, useTxtFmt, &syn0[(toks_cnt+a) * size_gramm], 0, size_gramm);
        VectorsModel::write_embedding__fin(fo);
      }
      free(support_oov_embedding);
    }
    fclose(fo);
    free(stub);
  }
  void calc_support_embedding( size_t words_count, size_t emb_size, float* embeddings, float* support_embedding ) const
  {
    for (size_t d = 0; d < emb_size; ++d)
    {
      float lbound = 1e10;
      for (size_t w = 0; w < words_count; ++w)
      {
        float *offs = embeddings + w*emb_size + d;
        if ( *offs < lbound )
          lbound = *offs;
      }
      *(support_embedding + d) = lbound - 0.01; // добавляем немного, чтобы не растянуть пространство
    }
  } // method-end
//  inline void softmax(float* uVec, size_t sz)
//  {
//    float max = std::numeric_limits<float>::min();
//    float sum = 0.0;
//    for (size_t i = 0; i < sz; ++i)
//      if (max < uVec[i])
//        max = uVec[i];
//    for (size_t i = 0; i < sz; ++i)
//    {
//      uVec[i] = std::exp(uVec[i] - max);   // overflow guard             TODO: использовать предобсчитанную таблицу экспонент с лимитами?
//      sum += uVec[i];
//    }
//    for (size_t i = 0; i < sz; ++i)
//      uVec[i] /= sum;
//  }
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
    VectorsModel vm;
    if ( !vm.load(filename, useTxtFmt) )
      return;
    if (vm.emb_size != layer1_size) { std::cerr << "Append: Dimensions fail" << std::endl; return; }
    // сохраняем старую модель, затем текущую
    FILE *fo = fopen(filename.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", vm.words_count + w_vocabulary->size(), layer1_size);
    for (size_t a = 0; a < vm.vocab.size(); ++a)
      VectorsModel::write_embedding(fo, useTxtFmt, vm.vocab[a], &vm.embeddings[a * vm.emb_size], vm.emb_size);
    if ( !useTxtFmt )
      saveEmbeddingsBin_helper(fo, w_vocabulary, syn0, layer1_size);
    else
      saveEmbeddingsTxt_helper(fo, w_vocabulary, syn0, layer1_size);
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
    }
    start_learning_tp = std::chrono::steady_clock::now();
    return true;
  } // method-end
  // функция восстановления ассоциативной весовой матрицы по векторной модели
  bool restore_assoc_by_model(const VectorsModel& vm)
  {
    if (!assoc_ctx_vocabulary)
      return true;
    for (size_t a = 0; a < assoc_ctx_vocabulary->size(); ++a)
    {
      auto& aword = assoc_ctx_vocabulary->idx_to_data(a).word;
      size_t w_idx = vm.get_word_idx(aword);
      if (w_idx == vm.words_count)
      {
        std::cerr << "restore_assoc_by_model: vocabs inconsistency" << std::endl;
        return false;
      }
      float* assoc_offset = vm.embeddings + w_idx * vm.emb_size + size_dep;
      float* trg_offset = syn1_assoc + a * size_assoc;
      std::copy(assoc_offset, assoc_offset + size_assoc, trg_offset);
    }
    return true;
  } // method-end
  // функция восстановления левой весовой матрицы из векторной модели
  bool restore_left_matrix_by_model(const VectorsModel& vm)
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
  // размерность части эмбеддинга, обучаемого грамматическим признакам
  size_t size_gramm;
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
  int *table_dep = nullptr;

  // вычисление очередного случайного значения (для случайного выбора векторов в рамках процедуры negative sampling)
  inline void update_random_ns(unsigned long long& next_random_ns)
  {
    next_random_ns = next_random_ns * (unsigned long long)25214903917 + 11;
  }
  // функция инициализации распределения, имитирующего шум, для метода оптимизации negative sampling
  void InitUnigramTable(int*& table, std::shared_ptr< CustomVocabulary > vocabulary)
  {
    // таблица униграм, посчитанная на основе частот слов с учетом имитации сабсэмплинга
    double norma = 0;
    double d1 = 0;
    table = (int *)malloc(table_size * sizeof(int));
    // вычисляем нормирующую сумму (с учётом сабсэмплинга)
    for (size_t a = 0; a < vocabulary->size(); ++a)
      norma += vocabulary->idx_to_data(a).cn * vocabulary->idx_to_data(a).sample_probability;
    // заполняем таблицу распределения, имитирующего шум
    size_t i = 0;
    d1 = vocabulary->idx_to_data(i).cn * vocabulary->idx_to_data(i).sample_probability / norma;
    for (size_t a = 0; a < table_size; ++a)
    {
      table[a] = i;
      if (a / (double)table_size > d1)
      {
        i++;
        d1 += vocabulary->idx_to_data(i).cn * vocabulary->idx_to_data(i).sample_probability / norma;
      }
      if (i >= vocabulary->size())
        i = vocabulary->size() - 1;
    }
  } // method-end
  // функция, реализующая модель обучения skip-gram
  void skip_gram( const LearningExample& le, float *neu1e, unsigned long long& next_random_ns )
  {
    size_t selected_ctx;   // хранилище для индекса контекста
    int label;             // метка класса; знаковое целое (!)
    float g = 0;           // хранилище для величины ошибки
    // вычисляем смещение вектора, соответствующего целевому слову
    float *targetVectorPtr = syn0 + le.word * layer1_size;
    // цикл по синтаксическим контекстам
    for (auto&& ctx_idx : le.dep_context)
    {
      // зануляем текущие значения ошибок (это частная производная ошибки E по выходу скрытого слоя h)
      std::fill(neu1e, neu1e+size_dep, 0.0);
      for (size_t d = 0; d <= negative; ++d)
      {
        if (d == 0) // на первой итерации рассматриваем положительный пример (контекст)
        {
          selected_ctx = ctx_idx;
          label = 1;
        }
        else // на остальных итерациях рассматриваем отрицательные примеры (случайные контексты из noise distribution)
        {
          update_random_ns(next_random_ns);
          selected_ctx = table_dep[(next_random_ns >> 16) % table_size];
          label = 0;
        }
        // вычисляем смещение вектора, соответствующего очередному положительному/отрицательному примеру
        float *ctxVectorPtr = syn1_dep + selected_ctx * size_dep;
        // в skip-gram выход скрытого слоя в точности соответствует вектору целевого слова
        // вычисляем выход нейрона выходного слоя (нейрона, соответствующего рассматриваемому положительному/отрицательному примеру) (hidden -> output)
        float f = std::inner_product(targetVectorPtr, targetVectorPtr+size_dep, ctxVectorPtr, 0.0);
        if ( std::isnan(f) ) continue;
        f = sigmoid(f);
        // вычислим ошибку, умноженную на коэффициент скорости обучения
        g = (label - f) * alpha;
        // обратное распространение ошибки output -> hidden
        if (d==0)
          std::transform(neu1e, neu1e+size_dep, ctxVectorPtr, neu1e, [g](float a, float b) -> float {return a + g*b;});
        else
        {
          float g_norm = g / negative;
          std::transform(neu1e, neu1e+size_dep, ctxVectorPtr, neu1e, [g_norm](float a, float b) -> float {return a + g_norm*b;});
        }
        // обучение весов hidden -> output
        if ( !proper_names )
          std::transform(ctxVectorPtr, ctxVectorPtr+size_dep, targetVectorPtr, ctxVectorPtr, [g](float a, float b) -> float {return a + g*b;});
      } // for all samples
      // обучение весов input -> hidden
      std::transform(targetVectorPtr, targetVectorPtr+size_dep, neu1e, targetVectorPtr, std::plus<float>());
    } // for all dep contexts

    // цикл по ассоциативным контекстам
    targetVectorPtr += size_dep; // используем оставшуюся часть вектора для ассоциаций
    if (!proper_names)
    {
      for (auto&& ctx_idx : le.assoc_context)
      {
        for (size_t d = 0; d <= negative; ++d)
        {
          if (d == 0) // на первой итерации рассматриваем положительный пример (контекст)
          {
            selected_ctx = ctx_idx;
            label = 1;
          }
          else // на остальных итерациях рассматриваем отрицательные примеры (случайные контексты из noise distribution)
          {
            update_random_ns(next_random_ns);
            selected_ctx = (next_random_ns >> 16) % w_vocabulary_size; // uniform distribution
            // отталкиваем даже стоп-слова!
            label = 0;
          }
          // вычисляем смещение вектора, соответствующего очередному положительному/отрицательному примеру
          float *ctxVectorPtr = syn0 + selected_ctx * layer1_size + size_dep;
          // вычисляем выход нейрона выходного слоя (нейрона, соответствующего рассматриваемому положительному/отрицательному примеру) (hidden -> output)
          float f = std::inner_product(targetVectorPtr, targetVectorPtr+size_assoc, ctxVectorPtr, 0.0);
          if ( std::isnan(f) ) continue;
          f = sigmoid(f);
          // вычислим ошибку, умноженную на коэффициент скорости обучения
          g = (label - f) * alpha;
          // обучение весов (input only)
          if (d == 0)
            std::transform(targetVectorPtr, targetVectorPtr+size_assoc, ctxVectorPtr, targetVectorPtr, [g](float a, float b) -> float {return a + g*b;});
          else
            std::transform(ctxVectorPtr, ctxVectorPtr+size_assoc, targetVectorPtr, ctxVectorPtr, [g](float a, float b) -> float {return a + g*b;});
        } // for all samples
      } // for all assoc contexts
    }
    else
    {
      for (auto&& ctx_idx : le.assoc_context)
      {
        float *ctxVectorPtr = syn1_assoc + ctx_idx * size_assoc;
        float f = std::inner_product(targetVectorPtr, targetVectorPtr+size_assoc, ctxVectorPtr, 0.0);
        if ( std::isnan(f) ) continue;
        f = sigmoid(f);
        g = (1.0 - f) * alpha;
        std::transform(targetVectorPtr, targetVectorPtr+size_assoc, ctxVectorPtr, targetVectorPtr, [g](float a, float b) -> float {return a + g*b;});
      } // for all assoc contexts
    }
  } // method-end

  // вычисление значения сигмоиды
  inline float sigmoid(float f) const
  {
    if      (f > MAX_EXP)  return 1;
    else if (f < -MAX_EXP) return 0;
    else                   return expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
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
      VectorsModel::write_embedding(fo, false, vocabulary->idx_to_data(a).word, &weight_matrix[a * emb_size], emb_size);
  } // method-end
  void saveEmbeddingsTxt_helper(FILE *fo, std::shared_ptr< CustomVocabulary > vocabulary, float *weight_matrix, size_t emb_size) const
  {
    for (size_t a = 0; a < vocabulary->size(); ++a)
      VectorsModel::write_embedding(fo, true, vocabulary->idx_to_data(a).word, &weight_matrix[a * emb_size], emb_size);
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
}; // class-decl-end


#endif /* TRAINER_H_ */
