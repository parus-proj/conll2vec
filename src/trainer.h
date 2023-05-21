#ifndef TRAINER_H_
#define TRAINER_H_

#include "learning_example_provider.h"
#include "vocabulary.h"
#include "original_word2vec_vocabulary.h"
#include "vectors_model.h"
#include "special_toks.h"

#include <memory>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>

#include "log.h"

#ifdef _MSC_VER
  #define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)
  #define free_aligned(p) _aligned_free((p))
#else
  #define free_aligned(p) free((p))
#endif


//#define EXP_TABLE_SIZE 1000
//#define MAX_EXP 6
#define EXP_TABLE_SIZE 2500
#define MAX_EXP 15


// forward decls
class ThreadsInWorkCounterGuard;


// хранит общие параметры и данные для всех потоков
// реализует логику обучения
class Trainer
{
  friend class ThreadsInWorkCounterGuard;
public:
  // конструктор
  Trainer( std::shared_ptr< LearningExampleProvider> learning_example_provider,
           std::shared_ptr< CustomVocabulary > words_vocabulary,
           bool trainTokens,
           std::shared_ptr< CustomVocabulary > dep_contexts_vocabulary,
           std::shared_ptr< CustomVocabulary > assoc_contexts_vocabulary,
           size_t embedding_dep_size,
           size_t embedding_assoc_size,
           size_t embedding_gramm_size,
           size_t epochs,
           float learning_rate,
           size_t negative_count_d,
           size_t negative_count_a,
           size_t total_threads_count )
  : lep(learning_example_provider)
  , w_vocabulary(words_vocabulary)
  , w_vocabulary_size(words_vocabulary->size())
  , toks_train(trainTokens)
  , dep_ctx_vocabulary(dep_contexts_vocabulary)
  , assoc_ctx_vocabulary(assoc_contexts_vocabulary)
  , layer1_size(embedding_dep_size + embedding_assoc_size)
  , size_dep(embedding_dep_size)
  , size_assoc(embedding_assoc_size)
  , size_gramm(embedding_gramm_size)
  , epoch_count(epochs)
  , alpha(learning_rate)
  , starting_alpha(learning_rate)
  , negative_d(negative_count_d)
  , negative_a(negative_count_a)
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

//    dbg_id1 = w_vocabulary->word_to_idx("судов");
//    dbg_id2 = w_vocabulary->word_to_idx("судно");
//    dbg_id3 = w_vocabulary->word_to_idx("суд");
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
    std::unique_ptr<ThreadsInWorkCounterGuard> wth_guard = std::make_unique<ThreadsInWorkCounterGuard>(this);

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
          if (dep_se_cnt >= 1000)
            do_sync_action(thread_idx, &Trainer::rescale_dep);
          if (ass_se_cnt >= 1000000)
            do_sync_action(thread_idx, &Trainer::rescale_assoc);
          if ((upd_ss_cnt == 0 && fraction >= 0.25) || (upd_ss_cnt == 1 && fraction >= 0.5) || (upd_ss_cnt == 2 && fraction >= 0.75))
            do_sync_action(thread_idx, &Trainer::decrease_subsampling);
        } // if ('checkpoint')
        // читаем очередной обучающий пример
        auto learning_example = lep->get(thread_idx, fraction);
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
        auto learning_example = lep->get(thread_idx, fraction, true);
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
        // ограничение степени выраженности признака
        std::transform(wordVectorPtr, wordVectorPtr+size_gramm, wordVectorPtr, Trainer::space_threshold_functor);

      } // for all learning examples
      word_count_actual += (word_count - last_word_count);
      if ( !lep->epoch_unprepare(thread_idx) )
        return;
    } // for all epochs
    free(y);
    free(eo);
    free(eh);
  } // method-end: train_entry_point__gramm
  void saveGrammaticalEmbeddings(const VectorsModel& vm, float g_ratio, const std::string& oov_voc_fn, const std::string& filename) const
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
    size_t quasiNumIdx = w_vocabulary->word_to_idx("одиннадцать");
    FILE *fo = fopen(filename.c_str(), "wb");
    fprintf(fo, "%lu %lu %lu %lu %lu\n", vm.vocab.size() + (v_oov ? v_oov->size() : 0), vm.emb_size + size_gramm, vm.dep_size, vm.assoc_size, size_gramm);
    for (size_t a = 0; a < vm.vocab.size(); ++a)
    {
      VectorsModel::write_embedding__start(fo, vm.vocab[a]);
      VectorsModel::write_embedding__vec(fo, &vm.embeddings[a*vm.emb_size], 0, vm.emb_size);
      size_t tok_idx = w_vocabulary->word_to_idx(vm.vocab[a]);
      if ( tok_idx != INVALID_IDX )
        VectorsModel::write_embedding__vec(fo, &syn0[tok_idx * size_gramm], 0, size_gramm);
      else
      {
        if ( vm.vocab[a] == "@num@" && quasiNumIdx != vm.vocab.size() )
          VectorsModel::write_embedding__vec(fo, &syn0[quasiNumIdx * size_gramm], 0, size_gramm);
        else
          VectorsModel::write_embedding__vec(fo, stub, 0, size_gramm);
        //std::cout << "stub gramm part for: " << vm.vocab[a] << std::endl;       //todo: это из-за того, что леммы могут не попадать словарь токенов (по частотному порогу), а еще @num@ и пунктуация
      }
      VectorsModel::write_embedding__fin(fo);
    }
    if (v_oov)
    {
      // создаем опорный вектор для OOV (семантическая часть)
      float *support_oov_embedding = (float *) malloc(vm.emb_size*sizeof(float));
      //calc_support_embedding(vm.words_count, vm.emb_size, vm.embeddings, support_oov_embedding);
      auto toks_cnt = w_vocabulary->size() - v_oov->size();
      std::map<std::u32string, size_t> vm_vocab_spec;  // сконвертированный в u32string и частично фильтрованный vm.vocab (для оптимизации calc_support_embedding_oov)
      const size_t SFX_SOURCE_WORD_MIN_LEN = 6;
      for (size_t a = 0; a < vm.vocab.size(); ++a)
      {
        auto w32s = StrConv::To_UTF32(vm.vocab[a]);
        if (w32s.length() >= SFX_SOURCE_WORD_MIN_LEN )
          vm_vocab_spec[w32s] = a;
      }
      for (size_t a = 0; a < v_oov->size(); ++a)
      {
        //std::cout << "save: " << v_oov->idx_to_data(a).word << std::endl;
        calc_support_embedding_oov(vm, vm_vocab_spec, v_oov->idx_to_data(a).word, support_oov_embedding);
        VectorsModel::write_embedding__start(fo, v_oov->idx_to_data(a).word);
        VectorsModel::write_embedding__vec(fo, support_oov_embedding, 0, vm.emb_size);
        VectorsModel::write_embedding__vec(fo, &syn0[(toks_cnt+a) * size_gramm], 0, size_gramm);
        VectorsModel::write_embedding__fin(fo);
      }
      free(support_oov_embedding);
    }
    fclose(fo);
    free(stub);
  }
  void calc_support_embedding_oov(const VectorsModel& vm, const std::map<std::u32string, size_t>& vm_vocab_spec, const std::string& oov_rec, float* support_embedding) const
  {
    const std::u32string OOV = U"_OOV_";
    const size_t OOV_PREFIX_LEN = OOV.length();
    auto sfx = StrConv::To_UTF32(oov_rec).substr(OOV_PREFIX_LEN);
    std::map<std::u32string, size_t> vm_vocab_short;
    std::copy_if( vm_vocab_spec.begin(), vm_vocab_spec.end(), std::inserter(vm_vocab_short, vm_vocab_short.begin()),
                  [sfx](const std::pair<std::u32string, size_t>& v) { return v.first.rfind(sfx) == (v.first.length()-sfx.length()); }
                );
    for (size_t d = 0; d < vm.emb_size; ++d)
    {
      // воспользуемся методом Уэлфорда для вычисления среднего (чтобы избежать рисков переполнения при суммировании)
      // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
      float meanValue = 0;
      size_t mvc = 1;
      for (auto& vm_rec : vm_vocab_short)
      {
        float* offset = vm.embeddings + vm_rec.second*vm.emb_size + d;
        meanValue += ((*offset) - meanValue) / mvc;
        ++mvc;
      }
      *(support_embedding + d) = meanValue;
    }
  }
//  void calc_support_embedding( size_t words_count, size_t emb_size, float* embeddings, float* support_embedding ) const
//  {
//    for (size_t d = 0; d < emb_size; ++d)
//    {
//      float lbound = 1e10;
//      for (size_t w = 0; w < words_count; ++w)
//      {
//        float *offs = embeddings + w*emb_size + d;
//        if ( *offs < lbound )
//          lbound = *offs;
//      }
//      *(support_embedding + d) = lbound - 0.01; // добавляем немного, чтобы не растянуть пространство
//    }
//  } // method-end
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
  void saveEmbeddings(const std::string& filename, const VectorsModel* vm = nullptr) const
  {
    FILE *fo = fopen(filename.c_str(), "wb");
    size_t saving_vocab_size = w_vocabulary->size();
    if (toks_train && vm)
    {
      size_t specials_cnt = 0;
      for ( auto w : SPECIAL_TOKS )
        if ( vm->get_word_idx(w) != vm->vocab.size() )
          ++specials_cnt;
      saving_vocab_size += specials_cnt;
    }
    fprintf(fo, "%lu %lu %lu %lu %lu\n", saving_vocab_size, layer1_size, size_dep, size_assoc, size_gramm);
    if (toks_train && vm)
    {
      for ( auto w : SPECIAL_TOKS )
      {
        size_t swidx = vm->get_word_idx(w);
        if ( swidx != vm->vocab.size() )
          VectorsModel::write_embedding(fo, w, &vm->embeddings[swidx * layer1_size], layer1_size);
      }
    }
    saveEmbeddingsBin_helper(fo, w_vocabulary, syn0, layer1_size);
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
    // открываем файл резервной копии модели
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
  // функция усреднения векторов в векторном пространстве в соответствии с заданным списком
  // усреденный вектор записывается по идексу, соответствующему первому элементу списка
  void vectors_weighted_collapsing(const std::vector< std::vector< std::pair<size_t, float> > >& collapsing_info)
  {
    // выделение памяти для среднего вектора
    float *avg = (float *)calloc(layer1_size, sizeof(float));
    for (auto& group : collapsing_info)
    {
      std::fill(avg, avg+layer1_size, 0.0);
      for (auto& vec : group)
      {
        size_t idx = vec.first;
        float weight = vec.second;
        float *offset = syn0 + idx*layer1_size;
        for (size_t d = 0; d < layer1_size; ++d)
          *(avg+d) += *(offset+d) * weight;
      }
      float *offset = syn0 + group.front().first * layer1_size;
      std::copy(avg, avg+layer1_size, offset);
    }
    free(avg);
  } // method-end

  // вывод статистики о ходе обучения
  void print_training_stat() const
  {
    std::cout << std::endl << "Training statistics" << std::endl;
    std::cout << "Dep. sigmoid overflows: " << dep_se_total << std::endl;
    std::cout << "Assoc. sigmoid overflows: " << ass_se_total << std::endl;
  }

private:
  std::shared_ptr< LearningExampleProvider > lep;
  std::shared_ptr< CustomVocabulary > w_vocabulary;
  size_t w_vocabulary_size;
  bool toks_train;    // признак того, что тренируются словоформы
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
  size_t negative_d;
  size_t negative_a;
  // матрицы весов между слоями input-hidden и hidden-output
  float *syn0 = nullptr, *syn1_dep = nullptr, *syn1_assoc = nullptr;
  // табличное представление логистической функции в области определения [-MAX_EXP; +MAX_EXP]
  float *expTable = nullptr;
  // noise distribution for negative sampling
  const size_t table_size = 1e8; // 100 млн.
  int *table_dep = nullptr;
  // счетчики "ошибок" точности вычисления сигмоиды
  size_t dep_se_cnt = 0;
  size_t ass_se_cnt = 0;
  size_t dep_se_total = 0;
  size_t ass_se_total = 0;
  // количество операций изменения subsampling
  size_t upd_ss_cnt = 0;

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
  // функтор ограничения пространства
  static float space_threshold_functor(float value)
  {
    const float FEATURE_VALUE_THRESHOLD = 3.0;
    if ( value > FEATURE_VALUE_THRESHOLD ) return FEATURE_VALUE_THRESHOLD;
    else if ( value < -FEATURE_VALUE_THRESHOLD ) return -FEATURE_VALUE_THRESHOLD;
    else return value;
  }
  // функция, реализующая модель обучения skip-gram
  void skip_gram( const LearningExample& le, float *neu1e, unsigned long long& next_random_ns )
  {
    size_t selected_ctx;   // хранилище для индекса контекста
    int label;             // метка класса; знаковое целое (!)
    float g = 0;           // хранилище для величины ошибки

    // вычисляем смещение вектора, соответствующего целевому слову
    float *targetVectorPtr = syn0 + le.word * layer1_size;
    float *targetDepPtr = targetVectorPtr;                                     // смещение категориальной части вектора
    float *targetDepEndPtr = targetDepPtr + size_dep;
    float *targetAssocPtr = targetVectorPtr + size_dep;                        // смещение ассоциативной части вектора
    float *targetAssocEndPtr = targetAssocPtr + size_assoc;

    // цикл по синтаксическим контекстам
    for (auto&& ctx_idx : le.dep_context)
    {
      // зануляем текущие значения ошибок (это частная производная ошибки E по выходу скрытого слоя h)
      std::fill(neu1e, neu1e+size_dep, 0.0);
      for (size_t d = 0; d <= negative_d; ++d)
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
        float f = std::inner_product(targetDepPtr, targetDepEndPtr, ctxVectorPtr, 0.0);
        if ( std::isnan(f) ) continue;
        f = sigmoid(f);
        if (f == 0.0 || f == 1.0)
          ++dep_se_cnt;
        // вычислим ошибку, умноженную на коэффициент скорости обучения
        g = (label - f) * alpha;
        if (g == 0) continue;
        // обратное распространение ошибки output -> hidden
        if (d==0)
          std::transform(neu1e, neu1e+size_dep, ctxVectorPtr, neu1e, [g](float a, float b) -> float {return a + g*b;});
        else
        {
          float g_norm = g / negative_d;
          std::transform(neu1e, neu1e+size_dep, ctxVectorPtr, neu1e, [g_norm](float a, float b) -> float {return a + g_norm*b;});
        }
        // обучение весов hidden -> output
        if ( !toks_train )
        {
          std::transform(ctxVectorPtr, ctxVectorPtr+size_dep, targetDepPtr, ctxVectorPtr, [g](float a, float b) -> float {return a + g*b;});
          // ограничиваем значение в векторах контекста
          // это необходимо для недопущения паралича обучения; sigmoid вычисляется с ограченной точностью, и если
          // векторное произведение станет слишком большим (маленьким), то sigmoid выдаст +1 (-1), что сделает градиент нулевым на положительном примере
          std::transform(ctxVectorPtr, ctxVectorPtr+size_dep, ctxVectorPtr, Trainer::space_threshold_functor);
        }
      } // for all samples
      // обучение весов input -> hidden
      std::transform(targetDepPtr, targetDepEndPtr, neu1e, targetDepPtr, std::plus<float>());
      // ограничение степени выраженности признака
      std::transform(targetDepPtr, targetDepEndPtr, targetDepPtr, Trainer::space_threshold_functor);
    } // for all dep contexts
//    // DBG-start
//    if (le.word == dbg_id1 || le.word == dbg_id2 || le.word == dbg_id3)
//    {
//      float *vector1Ptr = syn0 + dbg_id1 * layer1_size;
//      float *vector2Ptr = syn0 + dbg_id2 * layer1_size;
//      float dist = std::inner_product(vector1Ptr, vector1Ptr+size_dep, vector2Ptr, 0.0);
//      dist /= std::sqrt( std::inner_product(vector1Ptr, vector1Ptr+size_dep, vector1Ptr, 0.0) );
//      dist /= std::sqrt( std::inner_product(vector2Ptr, vector2Ptr+size_dep, vector2Ptr, 0.0) );
//      auto sp = (le.word == dbg_id1) ? "\t*" : "";
//      auto s = std::to_string(dist) + sp;
//      Log::getInstance()(s);
////      if (le.word == dbg_id1)
////      {
////        std::string bb;
////        for (auto&& ctx_idx : le.dep_context)
////          bb += " " + dep_ctx_vocabulary->idx_to_data(ctx_idx).word;
////        Log::getInstance()(bb);
////      }
//    }
//    // DBG-end

    // при доучивании токенов не трогаем ассоциативную часть
    // иначе "ассоциативная лексическая семантика" переучивается на "грамматическую/синтаксическую сочетаемость"
    if (toks_train)
      return;

    // цикл по ассоциативным контекстам
    for (auto&& ctx_idx : le.assoc_context)
    {
      for (size_t d = 0; d <= negative_a; ++d)
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
        float f = std::inner_product(targetAssocPtr, targetAssocEndPtr, ctxVectorPtr, 0.0);
        if ( std::isnan(f) ) continue;
        f = sigmoid(f);
        if (f == 0.0 || f == 1.0)
          ++ass_se_cnt;
        // вычислим ошибку, умноженную на коэффициент скорости обучения
        g = (label - f) * alpha;
        if (g == 0) continue;
        // обучение весов (input only)
        if (d == 0)
        {
          std::transform(targetAssocPtr, targetAssocEndPtr, ctxVectorPtr, targetAssocPtr, [g](float a, float b) -> float {return a + g*b;});
          // ограничение степени выраженности признака
          std::transform(targetAssocPtr, targetAssocEndPtr, targetAssocPtr, Trainer::space_threshold_functor);
        }
        else
        {
          std::transform(ctxVectorPtr, ctxVectorPtr+size_assoc, targetAssocPtr, ctxVectorPtr, [g](float a, float b) -> float {return a + g*b;});
          // ограничение степени выраженности признака
          std::transform(ctxVectorPtr, ctxVectorPtr+size_assoc, ctxVectorPtr, Trainer::space_threshold_functor);
        }
      } // for all samples
    } // for all assoc contexts

//    // обработка "надежных" ассоциативных пар
//    for ( size_t d = 0; d < le.rassoc.size(); ++d )
//    {
//      auto&& lera = le.rassoc[d];
//      float sim = std::get<2>(lera);
//      float *vector1Ptr = syn0 + std::get<0>(lera) * layer1_size + size_dep;
//      float *vector2Ptr = syn0 + std::get<1>(lera) * layer1_size + size_dep;
//      std::transform(vector1Ptr, vector1Ptr+size_assoc, vector2Ptr, neu1e, std::minus<float>());
//      float e_dist = std::sqrt( std::inner_product(neu1e, neu1e+size_assoc, neu1e, 0.0) );
//      if ( e_dist > 0.1)
//      {
//        std::transform(neu1e, neu1e+size_assoc, neu1e, [this,sim](float a) -> float {return a * alpha * 0.5 * sim;});
//        std::transform(vector2Ptr, vector2Ptr+size_assoc, neu1e, vector2Ptr, std::plus<float>());
//        std::transform(vector1Ptr, vector1Ptr+size_assoc, neu1e, vector1Ptr, std::minus<float>());
//      }
//    } // if reliable associatives

    // обработка данных от внешних словарей
    for ( size_t d = 0; d < le.ext_vocab_data.size(); ++d )
    {
      auto&& data = le.ext_vocab_data[d];

      float *vector1Ptr = syn0 + data.word1 * layer1_size + data.dims_from;
      float *vector2Ptr = syn0 + data.word2 * layer1_size + data.dims_from;
      size_t to_end = data.dims_to - data.dims_from + 1;
      std::transform(vector1Ptr, vector1Ptr+to_end, vector2Ptr, neu1e, std::minus<float>());
      float e_dist = std::sqrt( std::inner_product(neu1e, neu1e+to_end, neu1e, 0.0) );
      if ( e_dist > 0.1) // требуем, чтобы стягиваемые вектора хоть немного, но различались, т.к. слова-то всё ж разные
      {
        std::transform(neu1e, neu1e+to_end, neu1e, [this](float a) -> float {return a * alpha;});
        std::transform(vector2Ptr, vector2Ptr+to_end, neu1e, vector2Ptr, std::plus<float>());
        if ( data.algo == evaPairwise)
          std::transform(vector1Ptr, vector1Ptr+to_end, neu1e, vector1Ptr, std::minus<float>());
      }
    } // for all ext vocabs data

  } // method-end

  // вычисление значения сигмоиды
  inline float sigmoid(float f) const
  {
    if      (f > MAX_EXP)  return 1;
    else if (f < -MAX_EXP) return 0;
    else                   return expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
  } // method-end

private:
  // блок полей для управления синхронным (однопоточным) действием
  std::mutex mtx;
  std::condition_variable cv1, cv2;
  const size_t IMPOSSIBLE_THREAD_IDX = 1000000;
  size_t sync_task_thread = IMPOSSIBLE_THREAD_IDX; // номер потока, взявшегося за выполнение действия (первый прибежавший)
  bool action_in_progress = false; // признак того, что действие начало выполняться
  size_t working_threads = 0; // счетчик работающих потоков

  void do_sync_action(size_t thread_idx, std::function<void(Trainer&)> action)
  {
    // функция выполняет некое действие action над векторными пространствами в однопоточном режиме (остальные потоки уходят в ожидание)
    // здесь локализована логика синхронизации, полезная логика реализуется в action
    // !!!! должна быть гарантия, что к моменту первого вызова этой функции все рабочие потоки уже запустились (корректна величина working_threads)
    {
      std::lock_guard<std::mutex> lock(mtx);
      if (sync_task_thread == IMPOSSIBLE_THREAD_IDX)
      {
        sync_task_thread = thread_idx;
        action_in_progress = true;
      }
    }
    if (thread_idx == sync_task_thread)
    {
      // ждем ухода в спячку всех работающих потоков
      std::unique_lock<std::mutex> cv_lock(mtx);
      cv1.wait(cv_lock, [this]{return working_threads == 1;});
      // делаем полезную работу
      std::cout << std::endl << "Sync action begin" << std::endl;
      action(const_cast<Trainer&>(*this));
      std::cout << "Sync action end" << std::endl;
      // выполняем финализирующие действия и разблокируем все потоки
      // здесь mtx захвачен (после cv1.wait), можно безопасно работать с управляющими полями
      sync_task_thread = IMPOSSIBLE_THREAD_IDX;
      action_in_progress = false;
      cv2.notify_all();
    }
    else
    {
      std::unique_lock<std::mutex> cv_lock(mtx);
      --working_threads;
      cv1.notify_one();
      cv2.wait(cv_lock, [this]{return action_in_progress == false;});
      ++working_threads;
    }
  }
  // масштабирование пространства
  void rescale_dep()
  {
    std::cout << "Dep. rescale" << std::endl;
    for (size_t a = 0; a < w_vocabulary->size(); ++a)
      std::transform(syn0+a*layer1_size, syn0+a*layer1_size+size_dep, syn0+a*layer1_size, [](float v) -> float {return v*0.95;});
    for (size_t a = 0; a < dep_ctx_vocabulary->size(); ++a)
      std::transform(syn1_dep+a*size_dep, syn1_dep+a*size_dep+size_dep, syn1_dep+a*size_dep, [](float v) -> float {return v*0.99;});
    dep_se_total += dep_se_cnt;
    dep_se_cnt = 0;
  }
  void rescale_assoc()
  {
    std::cout << "Assoc. rescale" << std::endl;
    for (size_t a = 0; a < w_vocabulary->size(); ++a)
      std::transform(syn0+a*layer1_size+size_dep, syn0+a*layer1_size+size_dep+size_assoc, syn0+a*layer1_size+size_dep, [](float v) -> float {return v*0.95;});
    ass_se_total += ass_se_cnt;
    ass_se_cnt = 0;
  }
  // обновление subsampling-коэффициентов
  void decrease_subsampling()
  {
    ++upd_ss_cnt;
    std::cout << "Decrease subsampling" << std::endl;
    lep->update_subsampling_rates(0.5, 0.95, 0.95); // выполняем первым, т.к. InitUnigramTable зависит от уже вычисленных sample_probability в словарях
    if (table_dep)
      free(table_dep);
    if ( dep_ctx_vocabulary )
      InitUnigramTable(table_dep, dep_ctx_vocabulary);
  }

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
      VectorsModel::write_embedding(fo, vocabulary->idx_to_data(a).word, &weight_matrix[a * emb_size], emb_size);
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
//  size_t dbg_id1;
//  size_t dbg_id2;
//  size_t dbg_id3;
}; // class-decl-end



// RAII-класс для scope-обновления счетчика работающих потоков
class ThreadsInWorkCounterGuard
{
public:
  ThreadsInWorkCounterGuard(Trainer* t)
  : trainer(t)
  {
    std::lock_guard<std::mutex> lock(trainer->mtx);
    ++trainer->working_threads;
  }
  ~ThreadsInWorkCounterGuard()
  {
    {
      std::lock_guard<std::mutex> lock(trainer->mtx);
      --trainer->working_threads;
    }
    trainer->cv1.notify_one();
  }
private:
  Trainer* trainer;
};


#endif /* TRAINER_H_ */
